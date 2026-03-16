"""Training loop for behavioral cloning and VLA policies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from mimicplay.data.dataset import DemoDataset
from mimicplay.envs import make as make_env
from mimicplay.models.bc_policy import BCPolicy
from mimicplay.models.vla_policy import VLAPolicy
from mimicplay.training.scheduler import build_scheduler

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


def run_training(config_path: Path, resume_path: Path | None = None) -> None:
    """Entry point used by the CLI."""
    with config_path.open("r") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(cfg["env"])
    obs_shape = env.observation_shape
    n_actions = len(env.get_action_space())

    frame_stack = int(cfg.get("frame_stack", 4))
    in_channels = frame_stack * obs_shape[2]
    model_type = cfg.get("model", "bc")

    if model_type == "bc":
        model = BCPolicy(obs_shape=obs_shape, n_actions=n_actions, in_channels=in_channels)
    elif model_type == "vla":
        model = VLAPolicy(obs_shape=obs_shape, n_actions=n_actions, in_channels=in_channels)
    else:
        raise ValueError(f"Unknown model type '{model_type}'")

    model.to(device)

    demo_dir = Path(cfg["demo_dir"])
    dataset = DemoDataset(demo_dir=demo_dir, frame_stack=frame_stack, augment=bool(cfg["augmentation"]))
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 2)),
    )

    # Optional class weighting for imbalanced actions
    action_counts = torch.zeros(n_actions, dtype=torch.float32)
    for _obs, actions, _lang in DataLoader(dataset, batch_size=256):
        for a in actions:
            action_counts[int(a)] += 1.0
    class_weights = torch.ones(n_actions, dtype=torch.float32)
    nonzero = action_counts > 0
    class_weights[nonzero] = action_counts[nonzero].sum() / (len(action_counts) * action_counts[nonzero])
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    scheduler = build_scheduler(optimizer, max_epochs=int(cfg["epochs"]))

    run = None
    if wandb is not None and cfg.get("wandb_project"):
        run = wandb.init(project=cfg["wandb_project"], config=cfg, name=cfg.get("wandb_run_name"))

    best_loss = float("inf")
    checkpoint_dir = Path(cfg.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        epoch_loss = 0.0
        total = 0
        for obs, actions, languages in loader:
            obs = obs.to(device)
            actions = actions.to(device)

            optimizer.zero_grad()
            if model_type == "bc":
                logits = model(obs)
            else:
                assert isinstance(model, VLAPolicy)
                lang_emb = model.encode_language(list(languages))
                lang_emb = lang_emb.to(device)
                logits = model(obs, lang_emb)
            loss = criterion(logits, actions)
            loss.backward()
            optimizer.step()

            batch_size = actions.shape[0]
            epoch_loss += float(loss.item()) * batch_size
            total += batch_size

        scheduler.step()
        avg_loss = epoch_loss / max(1, total)
        print(f"Epoch {epoch}: train_loss={avg_loss:.4f}")

        if run is not None:
            run.log({"epoch": epoch, "train_loss": avg_loss})

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = checkpoint_dir / f"{model_type}_{cfg['env']}_best.pt"
            torch.save({"model_state": model.state_dict(), "config": cfg}, ckpt_path)

    if run is not None:
        run.finish()

