from __future__ import annotations

import csv
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    MultiStepLR,
    ReduceLROnPlateau,
)
from tqdm.auto import tqdm

from cgcnn.model import CrystalGraphConvNet


@dataclass
class AverageMeter:
    val: float = 0.0
    avg: float = 0.0
    sum: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1):
        self.val = float(value)
        self.sum += float(value) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


class Trainer:
    """
    Unified trainer for regression/classification tasks (train/val/test).

    Expected DataLoader batch format:
      ((atom_feature, edge_feature, edge_index, crystal_atom_idx), target, sample_ids)
    """

    def __init__(
        self,
        task: str = "regression",
        model: nn.Module | None = None,
        model_kwargs: dict[str, Any] | None = None,
        optimizer: str = "Adam",
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        momentum: float = 0.9,
        scheduler: str = "CosLR",
        scheduler_params: dict[str, Any] | None = None,
        criterion: nn.Module | None = None,
        device: torch.device | str = "auto",
        seed: int | None = None,
        max_grad_norm: float | None = None,
        ckpt_path: str | Path | None = None,
    ):
        self.model = model
        self.model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        if not scheduler:
            raise ValueError("scheduler must be specified.")
        self.scheduler_name = scheduler
        self.scheduler_params = {} if scheduler_params is None else scheduler_params
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: Any | None = None

        if seed is not None:
            self._set_seed(seed)

        self.device = self._resolve_device(device)
        self.task = task

        if criterion is not None:
            self.criterion = criterion
        else:
            if self.task == "regression":
                self.criterion = nn.MSELoss()
            else:
                self.criterion = nn.NLLLoss()

        self.max_grad_norm = max_grad_norm
        self.ckpt_path = Path(ckpt_path) if ckpt_path is not None else None

        if self.model is not None:
            self.model.to(self.device)
            self._build_optimizer()

        self.history: list[dict[str, float]] = []

        if self.task == "regression":
            self.best_main_metric = float("inf")  # lower is better (MAE)
        else:
            self.best_main_metric = float("-inf")  # higher is better (e.g., ACC)

    @staticmethod
    def _set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _resolve_device(device: torch.device | str):
        if isinstance(device, torch.device):
            return device

        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")

        return torch.device(device)

    def _build_model(self, train_loader: torch.utils.data.DataLoader):
        if self.model is not None:
            return

        first_batch = next(iter(train_loader))
        (_, edge_feature, _, _), _, _ = first_batch
        edge_feature_len = edge_feature.size(1)

        model_kwargs = dict(self.model_kwargs)
        model_kwargs.setdefault("classification", self.task == "classification")
        if model_kwargs["classification"] != (self.task == "classification"):
            raise ValueError(
                f"model_kwargs.classification must match task ({self.task!r})"
            )

        self.model = CrystalGraphConvNet(
            edge_feature_len=edge_feature_len,
            **model_kwargs,
        )
        self.model.to(self.device)

    def _build_optimizer(self):
        if self.model is None:
            raise ValueError("Model is not initialized.")

        opt = self.optimizer_name
        if opt == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        elif opt == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif opt == "AdamW":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif opt == "RAdam":
            self.optimizer = torch.optim.RAdam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            raise NotImplementedError(
                "optimizer must be one of 'SGD', 'Adam', 'AdamW', 'RAdam'"
            )

    def _build_scheduler(self, epochs: int | None = None):
        if self.optimizer is None:
            self.scheduler = None
            return

        params = dict(self.scheduler_params)
        name = self.scheduler_name
        if name in {"MultiStepLR", "multistep"}:
            params.setdefault("milestones", [100])
            params.setdefault("gamma", 0.1)
            self.scheduler = MultiStepLR(self.optimizer, **params)
        elif name in {"ExponentialLR", "exp"}:
            params.setdefault("gamma", 0.98)
            self.scheduler = ExponentialLR(self.optimizer, **params)
        elif name in {"CosineAnnealingLR", "CosLR", "Cos", "cos", "coslr"}:
            if "T_max" not in params:
                if epochs is None:
                    raise ValueError(
                        "epochs is required for CosLR default T_max=10*epochs. "
                        "Pass scheduler_params['T_max'] or call fit(..., epochs=...)."
                    )
                params["T_max"] = 10 * epochs
            params.setdefault("eta_min", 1e-2 * self.learning_rate)
            self.scheduler = CosineAnnealingLR(self.optimizer, **params)
        elif name in {"CosRestartLR", "cosrestart"}:
            params.setdefault("T_0", 10)
            params.setdefault("T_mult", 2)
            params.setdefault("eta_min", 1e-2 * self.learning_rate)
            self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, **params)
        elif name in {"ReduceLROnPlateau", "plateau"}:
            params.setdefault("mode", "min" if self.task == "regression" else "max")
            params.setdefault("factor", 0.5)
            params.setdefault("patience", 10)
            self.scheduler = ReduceLROnPlateau(self.optimizer, **params)
        else:
            raise NotImplementedError(f"Unsupported scheduler: {name}")

    def _ensure_initialized(
        self,
        train_loader: torch.utils.data.DataLoader,
        epochs: int | None = None,
        build_scheduler: bool = True,
    ):
        self._build_model(train_loader)
        if self.optimizer is None:
            self._build_optimizer()
        if build_scheduler and self.scheduler is None:
            self._build_scheduler(epochs=epochs)

    def parameter_count(
        self,
        train_loader: torch.utils.data.DataLoader | None = None,
        trainable_only: bool = False,
    ) -> int:
        """
        Return model parameter count.

        If model is lazily initialized, provide ``train_loader`` so the model can
        infer edge feature length from data and be built before counting.
        """
        if self.model is None:
            if train_loader is None:
                raise ValueError(
                    "Model is not initialized. Provide train_loader to build model "
                    "before counting parameters."
                )
            self._ensure_initialized(train_loader, build_scheduler=False)

        params = self.model.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def _main_metric_name(self):
        return "mae" if self.task == "regression" else "acc"

    def _move_batch_to_device(
        self,
        batch: tuple[Any, torch.Tensor, list[str]],
    ):
        (
            (atom_feature, edge_feature, edge_index, crystal_atom_idx),
            target,
            sample_ids,
        ) = batch

        atom_feature = atom_feature.to(self.device, non_blocking=True)
        edge_feature = edge_feature.to(self.device, non_blocking=True)
        edge_index = edge_index.to(self.device, non_blocking=True)
        crystal_atom_idx = [
            idx.to(self.device, non_blocking=True) for idx in crystal_atom_idx
        ]
        target = target.to(self.device, non_blocking=True)

        return (
            atom_feature,
            edge_feature,
            edge_index,
            crystal_atom_idx,
            target,
            sample_ids,
        )

    def _prepare_target(self, target: torch.Tensor):
        if self.task == "regression":
            return target.float()  # shape [batch_size, 1]

        return target.view(-1).long()

    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor):
        return self.criterion(pred, target)

    @staticmethod
    def mae(pred: torch.Tensor, target: torch.Tensor):
        return torch.mean(torch.abs(pred - target))

    def _compute_metrics(self, pred: torch.Tensor, target: torch.Tensor):
        if self.task == "regression":
            return {"mae": self.mae(pred.detach(), target).item()}
        return {}

    def _pred_for_export(self, pred: torch.Tensor):
        if self.task == "regression":
            return pred.detach().view(-1).cpu().tolist()
        return []

    def _run_epoch(
        self,
        loader: torch.utils.data.DataLoader,
        training: bool,
        return_preds: bool = False,
    ):
        self.model.train(training)

        loss_meter = AverageMeter()
        metric_meters: dict[str, AverageMeter] = {}

        all_ids: list[str] = []
        all_targets: list[float] = []
        all_preds: list[float] = []

        start = time.time()

        phase = "Train" if training else "Eval"
        pbar = tqdm(
            loader,
            desc=phase,
            leave=False,
            dynamic_ncols=True,
        )

        scheduler_step_batches: set[int] = set()
        is_cos_scheduler = isinstance(self.scheduler, CosineAnnealingLR)
        if training and is_cos_scheduler:
            n_batches = len(loader)
            if n_batches > 0:
                scheduler_step_batches = {
                    max(1, (k * n_batches) // 10) for k in range(1, 11)
                }

        for idx, batch in enumerate(pbar):
            (
                atom_feature,
                edge_feature,
                edge_index,
                crystal_atom_idx,
                target,
                sample_ids,
            ) = self._move_batch_to_device(batch)
            target = self._prepare_target(target)

            if training:
                if self.optimizer is None:
                    raise ValueError("Optimizer is not initialized.")
                self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(training):
                pred = self.model(
                    atom_feature, edge_feature, edge_index, crystal_atom_idx
                )
                loss = self._compute_loss(pred, target)

                if training:
                    loss.backward()

                    if self.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )

                    self.optimizer.step()

                    # CosLR: step 10 times per epoch (CHGNet-style).
                    if idx + 1 in scheduler_step_batches:
                        self.scheduler.step()

            bs = target.size(0)
            loss_meter.update(loss.detach().item(), bs)

            batch_metrics = self._compute_metrics(pred, target)
            for k, v in batch_metrics.items():
                if k not in metric_meters:
                    metric_meters[k] = AverageMeter()
                metric_meters[k].update(v, bs)

            postfix = {"loss": f"{loss_meter.avg:.4f}"}
            for k, meter in metric_meters.items():
                postfix[k] = f"{meter.avg:.4f}"
            pbar.set_postfix(postfix)

            if return_preds:
                all_ids.extend(sample_ids)
                all_targets.extend(target.detach().view(-1).cpu().tolist())
                all_preds.extend(self._pred_for_export(pred))

        out: dict[str, Any] = {"loss": loss_meter.avg, "time_sec": time.time() - start}
        for k, meter in metric_meters.items():
            out[k] = meter.avg

        if return_preds:
            out["sample_ids"] = all_ids
            out["targets"] = all_targets
            out["preds"] = all_preds

        return out

    def train_one_epoch(self, train_loader: torch.utils.data.DataLoader):
        return self._run_epoch(train_loader, training=True, return_preds=False)

    @torch.no_grad()
    def validate(self, val_loader: torch.utils.data.DataLoader):
        return self._run_epoch(val_loader, training=False, return_preds=False)

    @torch.no_grad()
    def test(
        self,
        test_loader: torch.utils.data.DataLoader,
        output_csv: str | Path | None = None,
    ) -> dict[str, Any]:
        metrics = self._run_epoch(loader=test_loader, training=False, return_preds=True)

        if output_csv is not None:
            output_csv = Path(output_csv)
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            with output_csv.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["sample_id", "target", "pred"])
                for sid, y, p in zip(
                    metrics["sample_ids"], metrics["targets"], metrics["preds"]
                ):
                    writer.writerow([sid, y, p])

        return metrics

    def _step_scheduler(self, monitor_value: float):
        if self.scheduler is None:
            return

        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(monitor_value)
        elif isinstance(self.scheduler, CosineAnnealingLR):
            # CosLR is stepped inside the training epoch.
            return
        else:
            self.scheduler.step()

    def _is_better(self, current: float, best: float):
        if self.task == "regression":
            return current < best
        return current > best

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        if self.ckpt_path is None:
            return

        self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_main_metric": self.best_main_metric,
            "task": self.task,
            "history": self.history,
        }
        if self.scheduler is not None and hasattr(self.scheduler, "state_dict"):
            state["scheduler"] = self.scheduler.state_dict()

        torch.save(state, self.ckpt_path)

        if is_best:
            best_path = self.ckpt_path.with_name("model_best.pth.tar")
            torch.save(state, best_path)

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int,
        print_freq: int = 1,
    ):
        main_metric_name = self._main_metric_name()
        self._ensure_initialized(train_loader, epochs=epochs, build_scheduler=True)
        train_start = datetime.now()

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_one_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            self._step_scheduler(val_metrics[main_metric_name])
            current_lr = (
                self.optimizer.param_groups[0]["lr"]
                if self.optimizer is not None
                else float("nan")
            )
            elapsed_delta = datetime.now() - train_start
            elapsed_sec = elapsed_delta.total_seconds()

            row: dict[str, float] = {
                "epoch": float(epoch),
                "train_loss": float(train_metrics["loss"]),
                "val_loss": float(val_metrics["loss"]),
                "lr": float(current_lr),
                "elapsed_sec": float(elapsed_sec),
            }
            for k, v in train_metrics.items():
                if k not in {"loss", "time_sec"}:
                    row[f"train_{k}"] = float(v)
            for k, v in val_metrics.items():
                if k not in {"loss", "time_sec"}:
                    row[f"val_{k}"] = float(v)

            self.history.append(row)

            current_metric = val_metrics[main_metric_name]
            is_best = self._is_better(current_metric, self.best_main_metric)
            if is_best:
                self.best_main_metric = current_metric

            self.save_checkpoint(epoch, is_best)

            if epoch % print_freq == 0:
                elapsed_hms = str(elapsed_delta).split(".")[0]
                print(
                    f"Epoch [{epoch}/{epochs}] "
                    f"train_loss={train_metrics['loss']:.4f} "
                    f"val_loss={val_metrics['loss']:.4f} "
                    f"train_{main_metric_name}={train_metrics[main_metric_name]:.4f} "
                    f"val_{main_metric_name}={val_metrics[main_metric_name]:.4f} "
                    f"lr={current_lr:.3e} "
                    f"elapsed={elapsed_hms}"
                )

        return self.history
