"""
train.py - GAN training logic
=============================
Trains the Generator and Discriminator from model.py on the MNIST dataset.

Run from the command line:
    python train.py                       # default: 30 epochs on CPU
    python train.py --epochs 50           # train longer
    python train.py --cuda                # use NVIDIA GPU if available
    python train.py --resume --epochs 60  # continue from last checkpoint

The same train() function is also used by the Streamlit app (Train tab).

Files written:
    models/generator.pth          - generator only (used by the app)
    models/checkpoint_last.pth    - full state (resume training)
    models/training_history.json  - loss values per epoch
    generated_images/training_samples/epoch_XXX.png - preview per epoch
"""

import argparse
import json
import os
import sys
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from model import (
    CHECKPOINT_PATH,
    DATA_DIR,
    DEFAULT_LATENT_DIM,
    GENERATOR_PATH,
    HISTORY_PATH,
    MODELS_DIR,
    SAMPLES_DIR,
    Discriminator,
    Generator,
    get_device,
)

DEFAULT_CONFIG = {
    "epochs": 30,         # how many passes over the 60,000 training images
    "batch_size": 128,    # images processed at once
    "lr": 0.0002,         # learning rate (standard value for GANs)
    "beta1": 0.5,         # Adam beta1 (0.5 is standard for GANs)
    "latent_dim": DEFAULT_LATENT_DIM,
    "use_cuda": False,    # CPU by default
    "resume": False,      # continue from models/checkpoint_last.pth
    "seed": 42,
}


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _atomic_save(obj, path):
    """Save to a temp file then rename, so a crash never leaves a broken file."""
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_history():
    """Read the saved loss history (returns None if missing/unreadable)."""
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def save_history(history):
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f)


def get_dataloader(batch_size):
    """Load MNIST (downloaded automatically the first time, free, no API key).

    Images are scaled from [0, 1] to [-1, 1] so they match the Generator's
    Tanh output range.
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    try:
        dataset = datasets.MNIST(
            root=str(DATA_DIR), train=True, download=True, transform=transform
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not load or download the MNIST dataset. Check your internet "
            f"connection for the first run. Details: {exc}"
        ) from exc

    # num_workers=0 is the most reliable setting on Windows.
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
    )


def _validate(cfg):
    if cfg["epochs"] < 1:
        raise ValueError("epochs must be at least 1.")
    if cfg["batch_size"] < 2:
        raise ValueError("batch_size must be at least 2.")
    if cfg["lr"] <= 0:
        raise ValueError("learning rate must be positive.")
    if cfg["latent_dim"] < 2:
        raise ValueError("latent_dim must be at least 2.")


# --------------------------------------------------------------------------
# Main training function
# --------------------------------------------------------------------------
def train(config=None, progress_callback=None):
    """Train the GAN and return the loss history.

    progress_callback(info: dict) is called regularly with keys:
        progress (0..1), epoch, epochs, g_loss, d_loss, history, message
    This lets the Streamlit app show a live progress bar.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    _validate(cfg)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device(cfg["use_cuda"])
    if cfg["use_cuda"] and device.type != "cuda":
        print("[warning] CUDA requested but not available. Using CPU instead.")
    print(f"Using device: {device}")

    torch.manual_seed(cfg["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg["seed"])

    loader = get_dataloader(cfg["batch_size"])
    n_batches = len(loader)

    # ---- resume from checkpoint if requested ----------------------------
    ckpt = None
    if cfg["resume"]:
        if CHECKPOINT_PATH.exists():
            try:
                ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
                cfg["latent_dim"] = int(ckpt["latent_dim"])  # must match saved model
            except Exception as exc:
                raise RuntimeError(
                    f"Could not read checkpoint ({exc}). Delete the files in "
                    "models/ and train from scratch."
                ) from exc
        else:
            print("[info] No checkpoint found, starting from scratch.")

    # ---- build models, loss, optimizers ---------------------------------
    generator = Generator(cfg["latent_dim"]).to(device)
    discriminator = Discriminator().to(device)

    # BCEWithLogitsLoss = sigmoid + binary cross-entropy (stable)
    criterion = nn.BCEWithLogitsLoss()
    betas = (cfg["beta1"], 0.999)
    opt_g = torch.optim.Adam(generator.parameters(), lr=cfg["lr"], betas=betas)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=cfg["lr"], betas=betas)

    history = {"epoch": [], "g_loss": [], "d_loss": []}
    start_epoch = 0
    fixed_noise = torch.randn(64, cfg["latent_dim"], device=device)

    if ckpt is not None:
        try:
            generator.load_state_dict(ckpt["generator_state"])
            discriminator.load_state_dict(ckpt["discriminator_state"])
            opt_g.load_state_dict(ckpt["opt_g_state"])
            opt_d.load_state_dict(ckpt["opt_d_state"])
            history = ckpt["history"]
            start_epoch = int(ckpt["epochs_trained"])
            fixed_noise = ckpt["fixed_noise"].to(device)
            print(f"Resumed from epoch {start_epoch}.")
        except Exception as exc:
            raise RuntimeError(f"Checkpoint is incompatible: {exc}") from exc

    if start_epoch >= cfg["epochs"]:
        msg = (f"Already trained for {start_epoch} epochs. "
               f"Increase --epochs above {start_epoch} to continue.")
        print(msg)
        if progress_callback:
            progress_callback({"progress": 1.0, "epoch": start_epoch,
                               "epochs": cfg["epochs"], "g_loss": None,
                               "d_loss": None, "history": history, "message": msg})
        return history

    total_epochs_this_run = cfg["epochs"] - start_epoch
    start_time = time.time()

    # ---- training loop ----------------------------------------------------
    for epoch in range(start_epoch + 1, cfg["epochs"] + 1):
        g_total, d_total = 0.0, 0.0

        for i, (real_imgs, _) in enumerate(loader):  # labels (digits) are NOT used
            real_imgs = real_imgs.to(device)
            bs = real_imgs.size(0)

            # Label smoothing: real = 0.9 instead of 1.0 keeps D from getting
            # over-confident, which makes training more stable.
            real_labels = torch.full((bs, 1), 0.9, device=device)
            fake_labels = torch.zeros((bs, 1), device=device)

            # ---- 1) Train the Discriminator ------------------------------
            # Goal: score real images high, fake images low.
            z = torch.randn(bs, cfg["latent_dim"], device=device)
            fake_imgs = generator(z)

            opt_d.zero_grad()
            loss_real = criterion(discriminator(real_imgs), real_labels)
            # detach(): do not update the Generator during the D step
            loss_fake = criterion(discriminator(fake_imgs.detach()), fake_labels)
            d_loss = loss_real + loss_fake
            d_loss.backward()
            opt_d.step()

            # ---- 2) Train the Generator ----------------------------------
            # Goal: fool D, i.e. make D label the fakes as "real".
            opt_g.zero_grad()
            g_loss = criterion(discriminator(fake_imgs), torch.ones((bs, 1), device=device))
            g_loss.backward()
            opt_g.step()

            g_total += g_loss.item()
            d_total += d_loss.item()

            # Report progress every 20 batches
            if progress_callback and (i % 20 == 0 or i == n_batches - 1):
                done = (epoch - start_epoch - 1) + (i + 1) / n_batches
                progress_callback({
                    "progress": min(done / total_epochs_this_run, 1.0),
                    "epoch": epoch, "epochs": cfg["epochs"],
                    "g_loss": g_loss.item(), "d_loss": d_loss.item(),
                    "history": history,
                    "message": f"Epoch {epoch}/{cfg['epochs']} - batch {i + 1}/{n_batches}",
                })

        # ---- end of epoch: record averages ------------------------------
        avg_g, avg_d = g_total / n_batches, d_total / n_batches
        history["epoch"].append(epoch)
        history["g_loss"].append(avg_g)
        history["d_loss"].append(avg_d)

        elapsed = time.time() - start_time
        print(f"Epoch [{epoch:3d}/{cfg['epochs']}]  D loss: {avg_d:.4f}  "
              f"G loss: {avg_g:.4f}  elapsed: {elapsed:.0f}s")

        # Preview image grid made from the SAME noise every epoch, so you can
        # watch the digits improve over time.
        generator.eval()
        with torch.no_grad():
            preview = generator(fixed_noise) * 0.5 + 0.5  # back to [0, 1]
        generator.train()
        save_image(preview, str(SAMPLES_DIR / f"epoch_{epoch:03d}.png"), nrow=8, padding=2)

        # ---- save checkpoints after every epoch --------------------------
        _atomic_save({
            "generator_state": generator.state_dict(),
            "latent_dim": cfg["latent_dim"],
            "epochs_trained": epoch,
        }, GENERATOR_PATH)
        _atomic_save({
            "generator_state": generator.state_dict(),
            "discriminator_state": discriminator.state_dict(),
            "opt_g_state": opt_g.state_dict(),
            "opt_d_state": opt_d.state_dict(),
            "latent_dim": cfg["latent_dim"],
            "epochs_trained": epoch,
            "history": history,
            "fixed_noise": fixed_noise.cpu(),
        }, CHECKPOINT_PATH)
        save_history(history)

        if progress_callback:
            progress_callback({
                "progress": (epoch - start_epoch) / total_epochs_this_run,
                "epoch": epoch, "epochs": cfg["epochs"],
                "g_loss": avg_g, "d_loss": avg_d, "history": history,
                "message": f"Finished epoch {epoch}/{cfg['epochs']}",
            })

    print(f"\nTraining complete. Generator saved to: {GENERATOR_PATH}")
    return history


# --------------------------------------------------------------------------
# Command-line entry point
# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train a simple GAN on MNIST.")
    p.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    p.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    p.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    p.add_argument("--latent-dim", type=int, default=DEFAULT_CONFIG["latent_dim"])
    p.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    p.add_argument("--cuda", action="store_true", help="use NVIDIA GPU if available")
    p.add_argument("--resume", action="store_true", help="continue from last checkpoint")
    return p.parse_args()


def main():
    args = parse_args()
    config = {
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "latent_dim": args.latent_dim, "seed": args.seed,
        "use_cuda": args.cuda, "resume": args.resume,
    }
    try:
        train(config)
    except KeyboardInterrupt:
        print("\nTraining interrupted. The last finished epoch is saved; "
              "use --resume to continue.")
        sys.exit(1)
    except (ValueError, RuntimeError) as exc:
        print(f"\n[error] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
