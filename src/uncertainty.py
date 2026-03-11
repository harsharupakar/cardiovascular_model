"""
uncertainty.py — Uncertainty estimation using Monte Carlo Dropout.
Performs multiple forward passes with dropout enabled to compute 
confidence intervals.
"""
import torch
import numpy as np

def mc_dropout_predict(model, X_tensor, n_passes=50, device="cpu"):
    """
    Performs n_passes forward passes through the model with dropout active.
    Returns mean probability and standard deviation (uncertainty).
    """
    model.train() # Enable dropout during inference
    probs_list = []
    
    X_tensor = X_tensor.to(device)
    
    with torch.no_grad():
        for _ in range(n_passes):
            logits = model(X_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
            probs_list.append(probs)
            
    probs_list = np.array(probs_list) # (n_passes, batch_size, 1)
    
    mean_prob   = np.mean(probs_list, axis=0).flatten()
    std_uncert  = np.std(probs_list,  axis=0).flatten()
    
    return mean_prob, std_uncert

def get_risk_with_confidence(prob, std, ci=1.96):
    """
    Returns a human-readable string with confidence interval.
    Example: "0.78 ± 0.07"
    """
    lower = max(0, prob - ci * std)
    upper = min(1, prob + ci * std)
    return f"{prob:.2f} ± {ci*std:.2f} (95% CI: [{lower:.2f}, {upper:.2f}])"
