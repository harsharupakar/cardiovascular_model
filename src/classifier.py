"""
classifier.py — MLP model definition, training loop with StratifiedKFold, 
and CTGAN augmentation (per-fold).
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, recall_score, f1_score
import mlflow

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import DATA_DIR, MODELS_DIR, SEED, set_seeds
from preprocess import preprocess
from ctgan_augment import augment_fold

set_seeds()

class CVDClassifier(nn.Module):
    def __init__(self, input_dim, hidden_size=128, dropout=0.3):
        super(CVDClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.net(x)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs)
            all_targets.extend(y_batch.numpy())
    return np.array(all_probs), np.array(all_targets)

def run_training(X, y, params, use_ctgan=True, n_folds=5, epochs=100, device="cpu"):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_results = []
    
    # Preprocess raw dataframe for CTGAN per-fold
    raw_df = pd.read_csv(os.path.join(DATA_DIR, "raw_dataset.csv"))
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold+1}/{n_folds} ---")
        
        # Data Preparation
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test   = X[test_idx], y[test_idx]
        
        if use_ctgan:
            # Per-fold CTGAN augmentation: train CTGAN on train_idx ONLY
            # We need the categorical/raw structure for CTGAN
            train_df_fold = raw_df.iloc[train_idx].copy()
            # Augment
            aug_df_fold = augment_fold(train_df_fold, n_synthetic=300, fold_idx=fold+1)
            # Re-preprocess augmented fold
            X_train, y_train, _, _ = preprocess(aug_df_fold, fit=False, 
                                               preprocessor=joblib.load(os.path.join(MODELS_DIR, "preprocessor.pkl")))

        # Convert to Tensors
        train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        test_ds  = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False)
        
        # Model, Criterion, Optimizer
        model = CVDClassifier(X.shape[1], hidden_size=params['hidden_size'], dropout=params['dropout']).to(device)
        
        # Pos weight for imbalance
        pos_count = np.sum(y_train)
        neg_count = len(y_train) - pos_count
        pos_weight = torch.tensor([neg_count / pos_count]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        
        best_auc = 0
        best_state = None
        
        for epoch in range(epochs):
            loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            probs, targets = evaluate(model, test_loader, device)
            auc = roc_auc_score(targets, probs)
            scheduler.step(loss)
            
            if auc > best_auc:
                best_auc = auc
                best_state = model.state_dict()
        
        print(f"  Best Fold AUC: {best_auc:.4f}")
        fold_results.append(best_auc)
    
    return np.mean(fold_results), np.std(fold_results)

import joblib
if __name__ == "__main__":
    # Placeholder for direct run
    from preprocess import preprocess
    raw_path = os.path.join(DATA_DIR, "raw_dataset.csv")
    df = pd.read_csv(raw_path)
    X, y, _, _ = preprocess(df)
    
    params = {'hidden_size': 128, 'dropout': 0.3, 'lr': 1e-3, 'weight_decay': 1e-4}
    mean_auc, std_auc = run_training(X, y, params, use_ctgan=False) # CTGAN off for quick test
    print(f"\nMean CV AUC: {mean_auc:.4f} +/- {std_auc:.4f}")
