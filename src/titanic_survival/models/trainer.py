"""
trainer.py — Loop de entrenamiento PyTorch profesional.

Implementa train_one_epoch + evaluate + fit con:
    - AdamW + CosineAnnealingLR
    - EarlyStopping con restore_best
    - Gradient clipping (clip_norm=1.0)
    - Mixed precision bfloat16 (RTX 4070)
    - Logging por época
"""

from __future__ import annotations

import logging
import time
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    clip_norm: float = 1.0,
    use_amp: bool = False,
) -> dict[str, float]:
    """Entrena el modelo durante un epoch completo."""
    model.train()
    tot_loss, n_correct, n_total = 0.0, 0, 0
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        yb_f = yb.float().unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(xb)
            loss = criterion(logits, yb_f)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        optimizer.step()

        preds = (torch.sigmoid(logits.detach()).squeeze(1) >= 0.5).long()
        n_correct += (preds == yb).sum().item()
        n_total += xb.size(0)
        tot_loss += loss.item() * xb.size(0)

    return {"loss": tot_loss / n_total, "acc": n_correct / n_total}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Evalúa el modelo sobre un DataLoader completo."""
    model.eval()
    tot_loss, n_correct, n_total = 0.0, 0, 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        yb_f = yb.float().unsqueeze(1)
        logits = model(xb)
        tot_loss += criterion(logits, yb_f).item() * xb.size(0)
        preds = (torch.sigmoid(logits).squeeze(1) >= 0.5).long()
        n_correct += (preds == yb).sum().item()
        n_total += xb.size(0)

    return {"loss": tot_loss / n_total, "acc": n_correct / n_total}


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    n_epochs: int = 80,
    scheduler: Any | None = None,
    clip_norm: float = 1.0,
    use_amp: bool = False,
    patience: int = 10,
    min_delta: float = 1e-4,
    seed: int = 42,
    log_every: int = 5,
) -> dict:
    """
    Orquesta el entrenamiento completo con EarlyStopping y restore_best.

    Returns
    -------
    dict
        Historial con train_loss, val_loss, train_acc, val_acc, lr por época.
    """
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    history: dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "lr": [],
    }

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    patience_count = 0
    stop_epoch = n_epochs

    for epoch in range(1, n_epochs + 1):
        t0 = time.perf_counter()
        tr = train_one_epoch(
            model, train_loader, optimizer, criterion, device, clip_norm=clip_norm, use_amp=use_amp
        )
        va = evaluate(model, val_loader, criterion, device)
        dt = time.perf_counter() - t0

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(va["loss"])
        elif scheduler is not None:
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(va["loss"])
        history["train_acc"].append(tr["acc"])
        history["val_acc"].append(va["acc"])
        history["lr"].append(lr)

        if epoch % log_every == 0 or epoch == 1:
            logger.info(
                "epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  " "val_acc=%.4f  lr=%.2e  (%.1fs)",
                epoch,
                n_epochs,
                tr["loss"],
                va["loss"],
                va["acc"],
                lr,
                dt,
            )

        # Early stopping
        if va["loss"] < best_val_loss - min_delta:
            best_val_loss = va["loss"]
            best_epoch = epoch
            patience_count = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= patience:
                stop_epoch = epoch
                logger.info(
                    "Early stop en epoch %d (mejor epoch=%d, val_loss=%.6f)",
                    epoch,
                    best_epoch,
                    best_val_loss,
                )
                break

    # Restaurar pesos de la mejor epoch
    if best_state is not None:
        model.load_state_dict(best_state)

    history["best_epoch"] = best_epoch
    history["stop_epoch"] = stop_epoch
    logger.info(
        "Entrenamiento completado. Mejor epoch=%d, val_loss=%.6f", best_epoch, best_val_loss
    )
    return history
