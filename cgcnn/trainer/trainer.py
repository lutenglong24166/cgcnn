from __future__ import annotations

import csv
import random
import time
from collections.abc import Mapping
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
        wandb_path: str | None = None,
        wandb_init_kwargs: dict[str, Any] | None = None,
        extra_run_config: dict[str, Any] | None = None,
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

        self.seed = seed
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
        self.wandb_path = wandb_path
        self.wandb_init_kwargs = (
            {} if wandb_init_kwargs is None else dict(wandb_init_kwargs)
        )
        self.extra_run_config = (
            {} if extra_run_config is None else dict(extra_run_config)
        )
        self._wandb: Any | None = None
        self._wandb_run: Any | None = None
        self.run_config: dict[str, Any] = {}

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
                        "Pass scheduler_params['T_max'] or call train(..., epochs=...)."
                    )
                params["T_max"] = 10 * epochs
            params.setdefault("eta_min", 1e-2 * self.learning_rate)
            self.scheduler = CosineAnnealingLR(self.optimizer, **params)
        elif name in {"CosRestartLR", "cosrestart"}:
            params.setdefault("T_0", 10)
            params.setdefault("T_mult", 2)
            params.setdefault("eta_min", 1e-2 * self.learning_rate)
            self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, **params)
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

    @staticmethod
    def _normalize_config_value(value: Any):
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Mapping):
            return {
                str(k): Trainer._normalize_config_value(v) for k, v in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [Trainer._normalize_config_value(v) for v in value]
        return str(value)

    @staticmethod
    def _parse_wandb_path(wandb_path: str):
        parts = wandb_path.split("/")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(
                "wandb_path must be in the format 'project/run_name', "
                f"got {wandb_path!r}"
            )
        return parts[0], parts[1]

    def _collect_run_config(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int,
        print_freq: int,
    ):
        config: dict[str, Any] = {
            "task": self.task,
            "seed": self.seed,
            "device": str(self.device),
            "optimizer": self.optimizer_name,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "momentum": self.momentum,
            "scheduler": self.scheduler_name,
            "scheduler_params": dict(self.scheduler_params),
            "max_grad_norm": self.max_grad_norm,
            "model_kwargs": dict(self.model_kwargs),
            "epochs": epochs,
            "print_freq": print_freq,
            "train_num_batches": len(train_loader),
            "val_num_batches": len(val_loader),
            "ckpt_path": str(self.ckpt_path) if self.ckpt_path is not None else None,
        }
        if self.model is not None:
            config["model_name"] = type(self.model).__name__
            config["parameter_count"] = sum(p.numel() for p in self.model.parameters())
        if self.wandb_path is not None:
            config["wandb_path"] = self.wandb_path

        config.update(self.extra_run_config)
        return self._normalize_config_value(config)

    def _maybe_init_wandb(self, run_config: dict[str, Any]):
        if self.wandb_path is None or self._wandb_run is not None:
            return

        try:
            import wandb  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "wandb is not installed. Install optional dependencies with "
                '`pip install -e ".[wandb]"`.'
            ) from exc

        project, run_name = self._parse_wandb_path(self.wandb_path)
        init_kwargs = dict(self.wandb_init_kwargs)
        user_config = init_kwargs.pop("config", None)

        merged_config = dict(run_config)
        if isinstance(user_config, Mapping):
            merged_config.update(user_config)
        elif user_config is not None:
            merged_config["user_config"] = self._normalize_config_value(user_config)

        init_kwargs.setdefault("project", project)
        init_kwargs.setdefault("name", run_name)
        init_kwargs["config"] = merged_config

        self._wandb = wandb
        self._wandb_run = wandb.init(**init_kwargs)

    def _wandb_log(self, metrics: dict[str, Any], step: int | None = None):
        if self._wandb_run is None or self._wandb is None:
            return
        self._wandb.log(metrics, step=step)

    def finish_wandb(self):
        if self._wandb_run is None or self._wandb is None:
            return
        self._wandb.finish()
        self._wandb_run = None

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
        pred_label = torch.argmax(pred.detach(), dim=1)
        acc = (pred_label == target).float().mean().item()
        return {"acc": acc}

    def _pred_for_export(self, pred: torch.Tensor):
        if self.task == "regression":
            return pred.detach().view(-1).cpu().tolist()
        pred_prob = torch.exp(pred.detach())
        if pred_prob.ndim != 2 or pred_prob.size(1) != 2:
            raise ValueError(
                "Classification export expects log-probabilities with shape [N, 2]."
            )
        return pred_prob[:, 1].cpu().tolist()

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
        if training and self.scheduler is not None:
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

                    # CHGNet-style: step scheduler 10 times per epoch.
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
        test_log = {
            f"test_{k}": float(v)
            for k, v in metrics.items()
            if k not in {"sample_ids", "targets", "preds"}
        }
        self._wandb_log(test_log, step=len(self.history) + 1)

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
            "run_config": self.run_config,
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
        wandb_log_freq: str = "epoch",
    ):
        if wandb_log_freq != "epoch":
            raise ValueError(
                "Only wandb_log_freq='epoch' is supported in this trainer."
            )

        main_metric_name = self._main_metric_name()
        self._ensure_initialized(train_loader, epochs=epochs, build_scheduler=True)
        self.run_config = self._collect_run_config(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            print_freq=print_freq,
        )
        self._maybe_init_wandb(self.run_config)
        train_start = datetime.now()

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_one_epoch(train_loader)
            val_metrics = self.validate(val_loader)

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

            row[f"best_val_{main_metric_name}"] = float(self.best_main_metric)
            self.save_checkpoint(epoch, is_best)
            self._wandb_log(row, step=epoch)

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
