"""
model.py - GAN architecture and shared helpers
==============================================
Synthetic Image Generation Using Deep Learning (MNIST + GAN)

This file contains:
  * Generator      : turns random noise into a 28x28 digit image
  * Discriminator  : decides whether a 28x28 image is real or fake
  * Helper functions for device selection and loading saved models

Both networks are simple fully-connected (MLP) networks, which keeps the
project easy to explain and fast to train even on a CPU.
"""

from pathlib import Path

import torch
from torch import nn

# --------------------------------------------------------------------------
# Project paths (all relative to this file, so it works from any folder)
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
GENERATED_DIR = BASE_DIR / "generated_images"
SAMPLES_DIR = GENERATED_DIR / "training_samples"   # one preview grid per epoch
DATA_DIR = BASE_DIR / "data"                       # MNIST is downloaded here

GENERATOR_PATH = MODELS_DIR / "generator.pth"          # light file used by the app
CHECKPOINT_PATH = MODELS_DIR / "checkpoint_last.pth"   # full state, for resuming
HISTORY_PATH = MODELS_DIR / "training_history.json"    # loss values per epoch

# --------------------------------------------------------------------------
# Image constants (MNIST images are 28 x 28 pixels, 1 grey channel)
# --------------------------------------------------------------------------
IMG_SIZE = 28
IMG_PIXELS = IMG_SIZE * IMG_SIZE  # 784
DEFAULT_LATENT_DIM = 100          # length of the random noise vector


def get_device(use_cuda: bool = False) -> torch.device:
    """Return the device to compute on.

    CPU is the default. CUDA (NVIDIA GPU) is only used when the caller asks
    for it AND a CUDA GPU is actually available.
    """
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Generator(nn.Module):
    """Generator: noise vector (latent_dim) -> fake image (1 x 28 x 28).

    Pixel values come out in the range [-1, 1] because of the final Tanh,
    which matches how the real images are normalised during training.
    """

    def __init__(self, latent_dim: int = DEFAULT_LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, IMG_PIXELS),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.net(z)
        return out.view(-1, 1, IMG_SIZE, IMG_SIZE)


class Discriminator(nn.Module):
    """Discriminator: image (1 x 28 x 28) -> a single "realness" score.

    The output is a raw score (logit). We use BCEWithLogitsLoss during
    training, which applies the sigmoid internally and is numerically stable.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(IMG_PIXELS, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        flat = img.view(img.size(0), -1)  # flatten 1x28x28 -> 784
        return self.net(flat)


def load_generator(path=GENERATOR_PATH, device=None):
    """Load a trained Generator from disk.

    Returns (generator, info_dict).
    Raises:
        FileNotFoundError : if the checkpoint does not exist (not trained yet)
        RuntimeError      : if the file is corrupted or incompatible
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained generator found at '{path}'. "
            "Run 'python train.py' first (or use the Train tab in the app)."
        )

    device = device or torch.device("cpu")
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
        latent_dim = int(ckpt["latent_dim"])
        generator = Generator(latent_dim).to(device)
        generator.load_state_dict(ckpt["generator_state"])
        generator.eval()  # inference mode
    except Exception as exc:  # corrupted / wrong format / version mismatch
        raise RuntimeError(
            f"The checkpoint '{path.name}' could not be loaded ({exc}). "
            "Delete it and train again."
        ) from exc

    info = {
        "latent_dim": latent_dim,
        "epochs_trained": int(ckpt.get("epochs_trained", 0)),
    }
    return generator, info
