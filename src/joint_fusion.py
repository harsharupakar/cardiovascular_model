"""
joint_fusion.py — Early/Joint multimodal fusion for CVD/HF risk.

This module is designed to prevent feature blindness seen in late-fusion setups
where critical echo findings are diluted by tabular lifestyle signals.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score


def _safe_index(name: str, names: List[str]) -> Optional[int]:
    try:
        return names.index(name)
    except ValueError:
        return None


def build_structural_severity_targets(
    x_struct: np.ndarray,
    struct_feature_names: List[str],
) -> np.ndarray:
    """
    Creates weak supervision labels for structural severity from echo features.

    Expected feature names (if available):
      - LVEF
      - LVEDD
      - WallMotion
      - MitralRegurgitation

    Returns binary targets in {0,1} with shape [N].
    """
    n = x_struct.shape[0]
    sev = np.zeros(n, dtype=np.float32)

    idx_lvef = _safe_index("LVEF", struct_feature_names)
    idx_lvedd = _safe_index("LVEDD", struct_feature_names)
    idx_wall = _safe_index("WallMotion", struct_feature_names)
    idx_mr = _safe_index("MitralRegurgitation", struct_feature_names)

    lvef = x_struct[:, idx_lvef] if idx_lvef is not None else np.full(n, 55.0, dtype=np.float32)
    lvedd = x_struct[:, idx_lvedd] if idx_lvedd is not None else np.full(n, 50.0, dtype=np.float32)
    wall = x_struct[:, idx_wall] if idx_wall is not None else np.zeros(n, dtype=np.float32)
    mr = x_struct[:, idx_mr] if idx_mr is not None else np.zeros(n, dtype=np.float32)

    severe_combo = (lvef <= 45.0) & ((lvedd >= 55.0) | (wall >= 1.0) | (mr >= 2.0))
    critical = (lvef <= 40.0) | (wall >= 2.0) | (mr >= 3.0)

    sev[severe_combo | critical] = 1.0
    return sev


class JointFusionHFNet(nn.Module):
    """
    Joint multimodal network:
      - Modality A: tabular/lifestyle EHR vector
      - Modality B: structural echo vector

    Key properties:
      1) Per-feature tokenization preserves identity of each variable.
      2) Self-attention learns intra-modality interactions.
      3) Bidirectional cross-attention learns cross-modal relationships.
      4) Explicit structural severity head prevents echo signal suppression.
      5) Learned fusion gate controls how much the model trusts structural data.
    """

    def __init__(
        self,
        tab_dim: int,
        struct_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        hidden: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.tab_dim = tab_dim
        self.struct_dim = struct_dim
        self.d_model = d_model

        self.tab_feature_embed = nn.Parameter(torch.randn(tab_dim, d_model) * 0.02)
        self.struct_feature_embed = nn.Parameter(torch.randn(struct_dim, d_model) * 0.02)
        self.tab_feature_bias = nn.Parameter(torch.zeros(tab_dim, d_model))
        self.struct_feature_bias = nn.Parameter(torch.zeros(struct_dim, d_model))

        self.tab_cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.struct_cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.tab_self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.struct_self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.tab_to_struct_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.struct_to_tab_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.norm_tab_1 = nn.LayerNorm(d_model)
        self.norm_struct_1 = nn.LayerNorm(d_model)
        self.norm_tab_2 = nn.LayerNorm(d_model)
        self.norm_struct_2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.struct_severity_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self.fusion_gate = nn.Sequential(
            nn.Linear(2 * d_model + 1, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model + 1, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def _build_tokens(self, x: torch.Tensor, feature_embed: torch.Tensor, feature_bias: torch.Tensor) -> torch.Tensor:
        """
        Convert scalar vector [B, F] to feature tokens [B, F, D].
        Each scalar scales a learned feature embedding, preserving per-feature identity.
        """
        return x.unsqueeze(-1) * feature_embed.unsqueeze(0) + feature_bias.unsqueeze(0)

    def forward(self, x_tab: torch.Tensor, x_struct: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with explicit multimodal fusion.

        x_tab:    [B, T]
        x_struct: [B, S]

        Returns:
          logits: [B] (for BCEWithLogitsLoss)
          extras: diagnostics for interpretability / auxiliary loss
        """
        # 1) Tokenize each modality separately.
        #    This preserves feature-level granularity for both tabular and echo inputs.
        tab_tokens = self._build_tokens(x_tab, self.tab_feature_embed, self.tab_feature_bias)            # [B, T, D]
        struct_tokens = self._build_tokens(x_struct, self.struct_feature_embed, self.struct_feature_bias) # [B, S, D]

        bsz = x_tab.size(0)
        tab_cls = self.tab_cls.expand(bsz, -1, -1)          # [B, 1, D]
        struct_cls = self.struct_cls.expand(bsz, -1, -1)    # [B, 1, D]

        # 2) Prepend CLS token per modality.
        #    CLS token acts as modality summary and is later used for fusion.
        tab_seq = torch.cat([tab_cls, tab_tokens], dim=1)         # [B, 1+T, D]
        struct_seq = torch.cat([struct_cls, struct_tokens], dim=1) # [B, 1+S, D]

        # 3) Intra-modality self-attention.
        #    Learns interactions inside each modality (e.g., BMI×glucose and LVEF×LVEDD).
        tab_self, _ = self.tab_self_attn(tab_seq, tab_seq, tab_seq)
        struct_self, _ = self.struct_self_attn(struct_seq, struct_seq, struct_seq)

        tab_seq = self.norm_tab_1(tab_seq + self.dropout(tab_self))
        struct_seq = self.norm_struct_1(struct_seq + self.dropout(struct_self))

        # 4) Bidirectional cross-attention.
        #    This is the key anti-blindness mechanism:
        #    - Tab query over structural keys/values forces tab representation to "see" echo state.
        #    - Structural query over tab keys/values lets structural interpretation be context-aware.
        tab_cross, tab_to_struct_attn = self.tab_to_struct_attn(
            query=tab_seq,
            key=struct_seq,
            value=struct_seq,
        )
        struct_cross, struct_to_tab_attn = self.struct_to_tab_attn(
            query=struct_seq,
            key=tab_seq,
            value=tab_seq,
        )

        tab_seq = self.norm_tab_2(tab_seq + self.dropout(tab_cross))
        struct_seq = self.norm_struct_2(struct_seq + self.dropout(struct_cross))

        # 5) Extract modality summaries from CLS tokens.
        tab_pooled = tab_seq[:, 0, :]       # [B, D]
        struct_pooled = struct_seq[:, 0, :] # [B, D]

        # 6) Explicit structural severity pathway.
        #    Provides a direct supervised route for critical echo anomalies so they cannot be
        #    completely diluted by low-risk lifestyle features.
        severity_logit = self.struct_severity_head(struct_pooled)  # [B, 1]
        severity_prob = torch.sigmoid(severity_logit)              # [B, 1]

        # 7) Learned modality gate.
        #    Gate uses both modality summaries + severity signal to adaptively weight structural input.
        gate_in = torch.cat([tab_pooled, struct_pooled, severity_prob], dim=1)  # [B, 2D+1]
        alpha = torch.sigmoid(self.fusion_gate(gate_in))                         # [B, 1]

        # 8) Final fused representation.
        #    alpha near 1 => rely more on structural signal.
        fused = (1.0 - alpha) * tab_pooled + alpha * struct_pooled               # [B, D]

        # 9) Final prediction combines fused latent + explicit severity score.
        #    This makes structural risk visible to the final classifier even if latent mixing is imperfect.
        classifier_in = torch.cat([fused, severity_prob], dim=1)                  # [B, D+1]
        logits = self.classifier(classifier_in).squeeze(1)                        # [B]

        extras = {
            "severity_logit": severity_logit.squeeze(1),
            "severity_prob": severity_prob.squeeze(1),
            "fusion_alpha": alpha.squeeze(1),
            "tab_to_struct_attn": tab_to_struct_attn,
            "struct_to_tab_attn": struct_to_tab_attn,
        }
        return logits, extras


def train_joint_fusion(
    model: JointFusionHFNet,
    x_tab_train: np.ndarray,
    x_struct_train: np.ndarray,
    y_train: np.ndarray,
    x_tab_val: np.ndarray,
    x_struct_val: np.ndarray,
    y_val: np.ndarray,
    struct_feature_names: List[str],
    device: str = "cpu",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    epochs: int = 80,
    patience: int = 10,
    aux_weight: float = 0.35,
) -> Dict[str, float]:
    """
    Train joint-fusion model with two losses:
      - Main loss: BCEWithLogits on CVD/HF label
      - Aux loss:  BCEWithLogits on structural severity weak label
    """
    model = model.to(device)

    sev_train = build_structural_severity_targets(x_struct_train, struct_feature_names)
    sev_val = build_structural_severity_targets(x_struct_val, struct_feature_names)

    train_ds = TensorDataset(
        torch.FloatTensor(x_tab_train),
        torch.FloatTensor(x_struct_train),
        torch.FloatTensor(y_train),
        torch.FloatTensor(sev_train),
    )
    val_ds = TensorDataset(
        torch.FloatTensor(x_tab_val),
        torch.FloatTensor(x_struct_val),
        torch.FloatTensor(y_val),
        torch.FloatTensor(sev_val),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Class imbalance handling for main target
    pos_count = max(float(np.sum(y_train)), 1.0)
    neg_count = max(float(len(y_train) - np.sum(y_train)), 1.0)
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32, device=device)

    main_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    aux_criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=4, factor=0.5)

    best_auc = 0.0
    best_state = None
    patience_counter = 0

    for _epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for xb_tab, xb_struct, yb, sb in train_loader:
            xb_tab = xb_tab.to(device)
            xb_struct = xb_struct.to(device)
            yb = yb.to(device)
            sb = sb.to(device)

            optimizer.zero_grad()
            logits, extras = model(xb_tab, xb_struct)

            main_loss = main_criterion(logits, yb)
            aux_loss = aux_criterion(extras["severity_logit"], sb)
            loss = main_loss + aux_weight * aux_loss

            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())

        train_loss = running_loss / max(len(train_loader), 1)

        model.eval()
        val_probs = []
        val_targets = []
        with torch.no_grad():
            for xb_tab, xb_struct, yb, _sb in val_loader:
                xb_tab = xb_tab.to(device)
                xb_struct = xb_struct.to(device)
                logits, _extras = model(xb_tab, xb_struct)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_probs.extend(probs)
                val_targets.extend(yb.numpy())

        val_probs_arr = np.array(val_probs)
        val_targets_arr = np.array(val_targets)
        val_auc = roc_auc_score(val_targets_arr, val_probs_arr)
        scheduler.step(train_loss)

        if val_auc > best_auc:
            best_auc = float(val_auc)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"best_val_auc": round(best_auc, 4)}


def predict_joint_fusion(
    model: JointFusionHFNet,
    x_tab: np.ndarray,
    x_struct: np.ndarray,
    device: str = "cpu",
) -> Dict[str, np.ndarray]:
    """
    Inference helper returning risk probabilities and fusion diagnostics.
    """
    model.eval()
    with torch.no_grad():
        tab_t = torch.FloatTensor(x_tab).to(device)
        struct_t = torch.FloatTensor(x_struct).to(device)
        logits, extras = model(tab_t, struct_t)
        probs = torch.sigmoid(logits).cpu().numpy()

    return {
        "probability": probs,
        "fusion_alpha": extras["fusion_alpha"].detach().cpu().numpy(),
        "severity_prob": extras["severity_prob"].detach().cpu().numpy(),
    }
