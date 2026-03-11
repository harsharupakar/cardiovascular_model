"""
calibration.py — Probability calibration using Temperature Scaling.
Evaluates Brier Score and saves calibration curves.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import OUTPUTS_DIR, MODELS_DIR

class TemperatureScaler(nn.Module):
    """
    A simple module to learn 'temperature' (T) to calibrate probabilities.
    Output = sigmoid(logits / T)
    """
    def __init__(self):
        super(TemperatureScaler, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

def calibrate_model(model, val_loader, device="cpu"):
    """
    Learns the temperature T on a validation set.
    """
    model.eval()
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            all_logits.append(logits)
            all_labels.append(y_batch)
            
    all_logits = torch.cat(all_logits).to(device)
    all_labels = torch.cat(all_labels).to(device).unsqueeze(1)
    
    scaler = TemperatureScaler().to(device)
    optimizer = optim.LBFGS(scaler.parameters(), lr=0.01, max_iter=50)
    
    criterion = nn.BCEWithLogitsLoss()
    
    def eval_loss():
        optimizer.zero_grad()
        loss = criterion(scaler(all_logits), all_labels)
        loss.backward()
        return loss
    
    optimizer.step(eval_loss)
    
    temp = scaler.temperature.item()
    print(f"Learned temperature: {temp:.4f}")
    return temp

def plot_calibration_curve(y_true, y_prob, name, out_path):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.plot(prob_pred, prob_true, "s-", label=f"{name}")
    
    plt.ylabel("Fraction of positives")
    plt.xlabel("Mean predicted probability")
    plt.ylim([-0.05, 1.05])
    plt.legend(loc="lower right")
    plt.title(f"Calibration Curve ({name})")
    plt.grid(True)
    plt.savefig(out_path)
    plt.close()
    print(f"Calibration curve saved to {out_path}")

def evaluate_calibration(y_true, y_prob):
    brier = brier_score_loss(y_true, y_prob)
    print(f"Brier Score: {brier:.4f}")
    return brier
