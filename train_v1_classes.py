# # BirdCLEF 2026 — EfficientNet-B1 (Class-based Dataset)
# 
# `BirdDataset` and `SoundscapeDataset` wrap PyTorch's `Dataset` interface. Each `__getitem__` loads, processes, and returns one sample; `DataLoader` handles batching, shuffling, and multi-worker prefetching automatically. The `BirdModel` class encapsulates the EfficientNet-B1 backbone and classification head. Phase 2 upgrade: change `CFG['backbone']` and load BirdSet weights into `model.backbone`.

# !pip install timm librosa -q

# ## Imports

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import timm

# ## Config

CFG = {
    'train_csv':        '/kaggle/input/birdclef-2026/train.csv',
    'soundscape_csv':   '/kaggle/input/birdclef-2026/train_soundscapes_labels.csv',
    'audio_dir':        '/kaggle/input/birdclef-2026/train_audio',
    'soundscape_dir':   '/kaggle/input/birdclef-2026/train_soundscapes',
    'taxonomy_csv':     '/kaggle/input/birdclef-2026/taxonomy.csv',
    'output_dir':       '/kaggle/working',
    'sample_rate':      32000,
    'duration':         5,
    'n_mels':           128,
    'n_fft':            1024,
    'hop_length':       320,
    'fmin':             20,
    'fmax':             16000,
    'backbone':         'efficientnet_b1',
    'pretrained':       True,
    'num_classes':      234,
    'fold':             0,
    'n_folds':          5,
    'epochs':           30,
    'batch_size':       32,
    'num_workers':      4,
    'lr':               1e-3,
    'weight_decay':     1e-4,
    'min_lr':           1e-6,
    'label_smoothing':  0.05,
    'mixup_alpha':      0.4,
    'specaug_freq_mask': 24,
    'specaug_time_mask': 64,
    'seed':             42,
    'device':           'cuda' if torch.cuda.is_available() else 'cpu',
}

np.random.seed(CFG['seed'])
torch.manual_seed(CFG['seed'])
torch.cuda.manual_seed_all(CFG['seed'])

# ## Labels

taxonomy  = pd.read_csv(CFG['taxonomy_csv'])
SPECIES   = taxonomy['primary_label'].tolist()
LABEL2IDX = {s: i for i, s in enumerate(SPECIES)}

# ## Datasets

class BirdDataset(Dataset):
    def __init__(self, df, audio_dir, cfg, train=True):
        self.df      = df.reset_index(drop=True)
        self.audio_dir = Path(audio_dir)
        self.cfg     = cfg
        self.train   = train
        self.samples = cfg['sample_rate'] * cfg['duration']

    def __len__(self): return len(self.df)

    def _load(self, path):
        wav, _ = librosa.load(path, sr=self.cfg['sample_rate'], mono=True)
        if len(wav) < self.samples:
            wav = np.pad(wav, (0, self.samples - len(wav)))
        start = np.random.randint(0, max(1, len(wav) - self.samples)) if self.train else 0
        return wav[start:start + self.samples]

    def _mel(self, wav):
        m = librosa.feature.melspectrogram(
            y=wav, sr=self.cfg['sample_rate'], n_mels=self.cfg['n_mels'],
            n_fft=self.cfg['n_fft'], hop_length=self.cfg['hop_length'],
            fmin=self.cfg['fmin'], fmax=self.cfg['fmax'],
        )
        m = librosa.power_to_db(m, ref=np.max)
        return ((m - m.min()) / (m.max() - m.min() + 1e-6)).astype(np.float32)

    def _augment(self, m):
        f, f0 = np.random.randint(0, self.cfg['specaug_freq_mask']), 0
        f0 = np.random.randint(0, max(1, m.shape[0] - f))
        m[f0:f0+f, :] = 0
        t  = np.random.randint(0, self.cfg['specaug_time_mask'])
        t0 = np.random.randint(0, max(1, m.shape[1] - t))
        m[:, t0:t0+t] = 0
        return m

    def _label(self, primary, secondary=''):
        label = np.zeros(self.cfg['num_classes'], dtype=np.float32)
        for sp in [primary] + (secondary.split() if isinstance(secondary, str) else []):
            sp = sp.strip("[],'\" ")
            if sp in LABEL2IDX: label[LABEL2IDX[sp]] = 1.0
        return label

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav = self._load(self.audio_dir / row['filename'])
        mel = self._mel(wav)
        if self.train: mel = self._augment(mel)
        x = torch.tensor(mel).unsqueeze(0).repeat(3, 1, 1)
        y = self._label(row['primary_label'], row.get('secondary_labels', ''))
        return x, torch.tensor(y)


class SoundscapeDataset(Dataset):
    def __init__(self, df, soundscape_dir, cfg):
        self.df      = df.reset_index(drop=True)
        self.sd      = Path(soundscape_dir)
        self.cfg     = cfg
        self.samples = cfg['sample_rate'] * cfg['duration']

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        wav, _ = librosa.load(self.sd / row['filename'], sr=self.cfg['sample_rate'],
                              mono=True, offset=int(row['start']), duration=self.cfg['duration'])
        if len(wav) < self.samples:
            wav = np.pad(wav, (0, self.samples - len(wav)))
        m = librosa.feature.melspectrogram(
            y=wav, sr=self.cfg['sample_rate'], n_mels=self.cfg['n_mels'],
            n_fft=self.cfg['n_fft'], hop_length=self.cfg['hop_length'],
            fmin=self.cfg['fmin'], fmax=self.cfg['fmax'],
        )
        m = librosa.power_to_db(m, ref=np.max)
        m = ((m - m.min()) / (m.max() - m.min() + 1e-6)).astype(np.float32)
        x = torch.tensor(m).unsqueeze(0).repeat(3, 1, 1)
        label = np.zeros(self.cfg['num_classes'], dtype=np.float32)
        for sp in str(row['primary_label']).split(';'):
            sp = sp.strip()
            if sp in LABEL2IDX: label[LABEL2IDX[sp]] = 1.0
        return x, torch.tensor(label)

# ## Model

class BirdModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = timm.create_model(
            cfg['backbone'], pretrained=cfg['pretrained'],
            num_classes=0, global_pool='avg',
        )
        in_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(0.3),
            nn.Linear(in_features, cfg['num_classes']),
        )

    def forward(self, x):
        return self.head(self.backbone(x))

# ## Loss, Mixup & Metric

def mixup_batch(x, y, alpha):
    if alpha <= 0:
        return x, y
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1-lam) * x[idx], lam * y + (1-lam) * y[idx]

class BCEWithLabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.05):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        targets = targets * (1 - self.smoothing) + self.smoothing / 2
        return self.bce(logits, targets)

def macro_roc_auc(targets, preds):
    scores = [
        roc_auc_score(targets[:, i], preds[:, i])
        for i in range(targets.shape[1]) if targets[:, i].sum() > 0
    ]
    return np.mean(scores) if scores else 0.0

# ## Train / Val Loops

def train_epoch(model, loader, optimizer, criterion, cfg):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(cfg['device']), y.to(cfg['device'])
        x, y = mixup_batch(x, y, cfg['mixup_alpha'])
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def val_epoch(model, loader, criterion, cfg):
    model.eval()
    total_loss, all_preds, all_targets = 0, [], []
    for x, y in loader:
        x, y = x.to(cfg['device']), y.to(cfg['device'])
        logits = model(x)
        total_loss += criterion(logits, y).item()
        all_preds.append(torch.sigmoid(logits).cpu().numpy())
        all_targets.append(y.cpu().numpy())
    return total_loss / len(loader), np.concatenate(all_preds), np.concatenate(all_targets)

# ## Run Training

train_df      = pd.read_csv(CFG['train_csv'])
soundscape_df = pd.read_csv(CFG['soundscape_csv'])

skf = StratifiedKFold(n_splits=CFG['n_folds'], shuffle=True, random_state=CFG['seed'])
train_df['fold'] = -1
for f, (_, vi) in enumerate(skf.split(train_df, train_df['primary_label'])):
    train_df.loc[vi, 'fold'] = f

fold   = CFG['fold']
tr_df  = train_df[train_df['fold'] != fold]
val_df = train_df[train_df['fold'] == fold]

tr_loader  = DataLoader(
    torch.utils.data.ConcatDataset([
        BirdDataset(tr_df, CFG['audio_dir'], CFG, train=True),
        SoundscapeDataset(soundscape_df, CFG['soundscape_dir'], CFG),
    ]),
    batch_size=CFG['batch_size'], shuffle=True, num_workers=CFG['num_workers'], pin_memory=True)
val_loader = DataLoader(
    BirdDataset(val_df, CFG['audio_dir'], CFG, train=False),
    batch_size=CFG['batch_size'], shuffle=False, num_workers=CFG['num_workers'], pin_memory=True)

model     = BirdModel(CFG).to(CFG['device'])
criterion = BCEWithLabelSmoothing(CFG['label_smoothing'])
optimizer = optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG['epochs'], eta_min=CFG['min_lr'])

best_auc = 0
out_path = Path(CFG['output_dir']) / f"best_fold{fold}.pt"

for epoch in range(CFG['epochs']):
    tr_loss              = train_epoch(model, tr_loader, optimizer, criterion, CFG)
    val_loss, preds, tgt = val_epoch(model, val_loader, criterion, CFG)
    auc                  = macro_roc_auc(tgt, preds)
    scheduler.step()
    print(f"Epoch {epoch+1:02d} | tr_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | AUC={auc:.4f}")
    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(), out_path)
        print(f"  -> saved (AUC={best_auc:.4f})")
