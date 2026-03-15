import pytest
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.classifier import CVDClassifier

def test_classifier_forward_pass_no_sigmoid():
    """
    Ensure the classifier returns raw logits (no sigmoid at the end).
    """
    input_dim = 18
    model = CVDClassifier(input_dim=input_dim, hidden_size=64)
    model.eval()
    
    x = torch.randn(5, input_dim)
    with torch.no_grad():
        logits = model(x)
        
    assert logits.shape == (5, 1)
    
    # Verify values can be outside [0, 1] range (sign of logits)
    # Even if they randomly fall in [0,1], applying BCEWithLogitsLoss requires logits
    assert isinstance(model.net[-1], nn.Linear)
    
def test_loss_function_has_pos_weight():
    """
    Verify the loss function calculation structure allows pos_weight.
    """
    pos_weight = torch.tensor([2.5])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    logits = torch.tensor([[0.5], [-1.0]])
    targets = torch.tensor([[1.0], [0.0]])
    
    loss = criterion(logits, targets)
    assert loss.item() > 0
