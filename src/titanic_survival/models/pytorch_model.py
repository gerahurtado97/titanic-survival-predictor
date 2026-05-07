"""
pytorch_model.py — MLP en PyTorch para clasificación binaria.

Implementa:
    TitanicDataset  — subclass de torch.utils.data.Dataset
    TitanicMLP      — subclass de nn.Module (≥ 3 capas ocultas, BatchNorm1d, GELU, Dropout)
    PytorchSurvivalModel — implementa BaseModel (interfaz común)

Arquitectura calibrada en Clase 8:
    Input(38) → 256×4 → 128×3 → 64 → 32 → 1 (logit)
    Activación: GELU (evita dead neurons, estándar en Transformers)
    pos_weight = n_neg/n_pos × 2.0 para maximizar Recall
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from titanic_survival.models.base import BaseModel

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset personalizado
# ─────────────────────────────────────────────────────────────────────────────


class TitanicDataset(Dataset):
    """
    Dataset PyTorch para el Titanic.

    Subclass de torch.utils.data.Dataset — obligatorio implementar
    __len__ y __getitem__ para que DataLoader funcione.

    Parameters
    ----------
    X : np.ndarray
        Features preprocesadas (shape: [N, IN_DIM]).
    y : np.ndarray | None
        Etiquetas (0/1). None en inferencia (sin etiquetas).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray | None = None) -> None:
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long) if y is not None else None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple:
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return (self.X[idx],)


# ─────────────────────────────────────────────────────────────────────────────
# Arquitectura del MLP
# ─────────────────────────────────────────────────────────────────────────────


class TitanicMLP(nn.Module):
    """
    MLP para clasificación binaria sobre datos tabulares del Titanic.

    Arquitectura por capa oculta: Linear → BatchNorm1d → GELU → Dropout

    Justificación de GELU sobre ReLU:
        Con N≈620 muestras, las neuronas muertas son un riesgo real.
        GELU tiene gradiente ≠ 0 para entradas ligeramente negativas,
        preserva la señal y es el estándar en BERT, GPT y LLaMA.

    La capa de salida devuelve un logit escalar (sin activación),
    pareja natural de BCEWithLogitsLoss.

    Parameters
    ----------
    in_dim : int
        Dimensión de entrada.
    hidden_dims : tuple[int, ...]
        Dimensión de cada capa oculta (mínimo 3).
    dropouts : tuple[float, ...]
        Dropout por capa (decreciente: más regularización en capas tempranas).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256, 256, 256, 128, 128, 128, 64, 32),
        dropouts: tuple[float, ...] = (0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2),
    ) -> None:
        super().__init__()

        if len(hidden_dims) < 3:
            raise ValueError(f"Se requieren ≥ 3 capas ocultas; recibido {len(hidden_dims)}")
        if len(hidden_dims) != len(dropouts):
            raise ValueError("hidden_dims y dropouts deben tener la misma longitud")

        layers: list[nn.Module] = []
        prev = in_dim
        for h, p in zip(hidden_dims, dropouts, strict=False):
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(p),
            ]
            prev = h

        layers.append(nn.Linear(prev, 1))  # logit de salida
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Devuelve logit de shape (batch, 1)."""
        return self.net(x)

    def count_parameters(self) -> int:
        """Número total de parámetros entrenables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper que implementa BaseModel
# ─────────────────────────────────────────────────────────────────────────────


class PytorchSurvivalModel(BaseModel):
    """
    Wrapper de TitanicMLP que implementa la interfaz BaseModel.

    Permite usar el modelo PyTorch de forma intercambiable con el
    modelo sklearn desde la app de Streamlit y los scripts de evaluación.
    """

    def __init__(
        self,
        in_dim: int = 38,
        hidden_dims: tuple[int, ...] = (256, 256, 256, 256, 128, 128, 128, 64, 32),
        dropouts: tuple[float, ...] = (0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2),
        device: str | None = None,
    ) -> None:
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.dropouts = dropouts
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._mlp: TitanicMLP | None = None

    def train(  # type: ignore[override]
        self,
        X: np.ndarray,
        y: np.ndarray,
        pos_weight_factor: float = 2.0,
        epochs: int = 80,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        clip_norm: float = 1.0,
        use_amp: bool = True,
        early_stopping_patience: int = 10,
        seed: int = 42,
    ) -> dict:
        """
        Entrena el MLP con AdamW + CosineAnnealingLR + EarlyStopping.

        Importa trainer internamente para evitar imports circulares.

        Returns
        -------
        dict
            Historial de entrenamiento (train_loss, val_loss, lr por epoch).
        """
        from titanic_survival.models.trainer import fit

        torch.manual_seed(seed)
        self._mlp = TitanicMLP(self.in_dim, self.hidden_dims, self.dropouts).to(self.device)

        # pos_weight para compensar desbalance + factor para maximizar Recall
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        pos_weight = torch.tensor([(n_neg / n_pos) * pos_weight_factor], dtype=torch.float32).to(
            self.device
        )

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(self._mlp.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=1e-6,
        )

        # Split train/val interno (80/20)
        n_val = max(1, int(0.20 * len(X)))
        idx = np.random.default_rng(seed).permutation(len(X))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]

        use_pin = self.device.type == "cuda"
        g = torch.Generator().manual_seed(seed)
        train_ds = TitanicDataset(X[tr_idx], y[tr_idx])
        val_ds = TitanicDataset(X[val_idx], y[val_idx])
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, generator=g, pin_memory=use_pin
        )
        val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, pin_memory=use_pin)

        history = fit(
            model=self._mlp,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=self.device,
            n_epochs=epochs,
            scheduler=scheduler,
            clip_norm=clip_norm,
            use_amp=use_amp and self.device.type == "cuda",
            patience=early_stopping_patience,
            seed=seed,
        )
        return history

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self._mlp is None:
            raise RuntimeError("Modelo no entrenado.")
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self._mlp is None:
            raise RuntimeError("Modelo no entrenado.")
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_t = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        self._mlp.eval()
        with torch.no_grad():
            logits = self._mlp(X_t)
            proba = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        return proba

    def save(self, path: str | Path) -> None:
        """Serializa el estado del MLP con torch.save()."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            torch.save(
                {
                    "model_state_dict": self._mlp.state_dict(),
                    "in_dim": self.in_dim,
                    "hidden_dims": self.hidden_dims,
                    "dropouts": self.dropouts,
                },
                fh,
            )
        size_kb = path.stat().st_size / 1024
        logger.info("Modelo PyTorch guardado: %s (%.1f KB)", path, size_kb)

    @classmethod
    def load(cls, path: str | Path) -> PytorchSurvivalModel:
        """Carga un modelo PyTorch desde disco."""
        path = Path(path)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        instance = cls(
            in_dim=ckpt["in_dim"],
            hidden_dims=tuple(ckpt["hidden_dims"]),
            dropouts=tuple(ckpt["dropouts"]),
        )
        instance._mlp = TitanicMLP(
            instance.in_dim,
            instance.hidden_dims,
            instance.dropouts,
        )
        instance._mlp.load_state_dict(ckpt["model_state_dict"])
        instance._mlp.to(instance.device)
        logger.info("Modelo PyTorch cargado desde: %s", path)
        return instance
