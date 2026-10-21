# Synthetic Image Generation Using Deep Learning

A simple **Generative Adversarial Network (GAN)** built with **PyTorch** and trained on the
**MNIST** handwritten digit dataset. A **Streamlit** web app lets you generate new digit images,
compare them with real ones, view training results, and download the images.

* 100% Python (no Node.js / React / Next.js)
* CPU by default, optional NVIDIA GPU (CUDA)
* No paid APIs and no API keys (MNIST is a free dataset)

## Project structure

```
synthetic-image-gan/
├── app.py              # Streamlit user interface
├── model.py            # Generator & Discriminator + helpers
├── train.py            # Training logic (command line + used by the app)
├── requirements.txt    # Python dependencies
├── README.md
├── .gitignore
├── models/             # saved checkpoints (created by training)
└── generated_images/   # saved outputs and per-epoch training previews
```

## Setup on Windows (step by step)

Requirements: **Python 3.9 - 3.12** installed (tick "Add Python to PATH" in the installer)
and internet access for the first run (to install packages and download MNIST).

**1. Open Command Prompt or PowerShell and go to the project folder**

```
cd path\to\synthetic-image-gan
```

**2. Create a virtual environment**

```
python -m venv venv
```

**3. Activate it**

Command Prompt (cmd):
```
venv\Scripts\activate.bat
```

PowerShell:
```
venv\Scripts\Activate.ps1
```

If PowerShell says "running scripts is disabled", run this once in the same window, then activate again:
```
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

You should now see `(venv)` at the start of the prompt.

**4. Install the requirements**

```
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**5. Train the GAN** (downloads MNIST the first time, about 11 MB)

```
python train.py
```

Default is 30 epochs on CPU (roughly 5 - 15 minutes). For a quick test use `python train.py --epochs 5`.

**6. Run the web app**

```
streamlit run app.py
```

Your browser opens at http://localhost:8501.

### Useful training options

| Command | Meaning |
|---|---|
| `python train.py --epochs 50` | train for 50 epochs |
| `python train.py --resume --epochs 60` | continue from the last checkpoint up to 60 epochs |
| `python train.py --cuda` | use the GPU if CUDA is available (falls back to CPU otherwise) |
| `python train.py --batch-size 256 --lr 0.0002` | change hyperparameters |

### Optional: GPU (CUDA) support

The default `pip install` gives a CPU build on Windows, which is enough for this project.
For an NVIDIA GPU, install a CUDA build of PyTorch instead (see https://pytorch.org for the right command), for example:

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Then run `python train.py --cuda` and tick **Use GPU (CUDA)** in the app sidebar.

## What the app contains

| Tab | What it does |
|---|---|
| Learn | Beginner-friendly explanation of GANs and synthetic images |
| Generate | Choose number of images (1-100), grid layout, seed; view, download grid / ZIP / single PNG; optionally save to `generated_images/` |
| Real vs Generated | Real MNIST digits next to GAN-generated digits |
| Training Results | Loss plot, metrics, and how digits improved epoch by epoch |
| Train | Train from the browser with a live progress bar and live loss chart |

## Files created by training

| File | Purpose |
|---|---|
| `models/generator.pth` | Generator weights (used by the app) |
| `models/checkpoint_last.pth` | Full state (both networks, optimizers, history) to resume training |
| `models/training_history.json` | Loss values per epoch |
| `generated_images/training_samples/epoch_XXX.png` | Preview grid after every epoch |
| `data/` | Downloaded MNIST dataset |

## Error handling

* **No trained model:** the app shows instructions instead of crashing.
* **Corrupted checkpoint:** a clear message tells you to delete it and retrain (checkpoints are written atomically to avoid this).
* **No internet on first run:** a readable error explains that MNIST could not be downloaded.
* **CUDA requested but not available:** automatically falls back to CPU with a warning.
* **Ctrl+C during training:** the last finished epoch is saved; continue with `--resume`.

## How it works (short)

1. **Generator** (100 random numbers -> 256 -> 512 -> 1024 -> 784 pixels, Tanh) creates a 28x28 image.
2. **Discriminator** (784 pixels -> 1024 -> 512 -> 256 -> 1 score) judges real vs fake.
3. Both are trained with **binary cross-entropy** and **Adam** (lr 0.0002, beta1 0.5).
4. Images are normalised to [-1, 1] to match the Generator's Tanh output.
5. Label smoothing (real label 0.9) and dropout in the Discriminator keep training stable.

## Viva questions and answers

**1. What is a GAN?** Two networks trained against each other: a Generator that creates fake data and a Discriminator that tells real from fake. Both improve through competition.

**2. What does the Generator take as input?** A random noise vector (100 values from a normal distribution). Different noise gives different images.

**3. Why do we use random noise?** It gives the Generator a source of variety, so each noise vector maps to a different new image. The same seed gives the same image.

**4. What loss function is used?** Binary cross-entropy (`BCEWithLogitsLoss`). The Discriminator wants real -> 1 and fake -> 0; the Generator wants fake -> 1.

**5. Why Tanh in the Generator and normalising to [-1, 1]?** Tanh outputs values between -1 and 1, so real images are scaled to the same range. Matching ranges makes learning easier.

**6. Why do the GAN losses not go to zero?** It is a two-player game. If one network wins completely, the other cannot learn. Stable, balanced losses are a good sign.

**7. Are the digit labels used?** No. A GAN is unsupervised; it learns what digits look like without knowing which digit is which.

**8. What is an epoch? A batch?** An epoch is one full pass over the 60,000 training images. A batch is the small group (128) processed in one step.

**9. What is mode collapse?** When the Generator produces only a few kinds of images (for example, only "1"s). Tricks like label smoothing and dropout reduce the risk.

**10. Why `detach()` in the Discriminator step?** So that gradients from the Discriminator loss do not update the Generator during that step.

**11. Applications of synthetic images?** Data augmentation, privacy-preserving data (medical images), art/design, super-resolution, simulation, and testing.

**12. Limitations of this project?** An MLP GAN on 28x28 grey images is simple; some digits look imperfect. A convolutional GAN (DCGAN) or longer training gives sharper results. There is also no control over which digit is produced (a conditional GAN could add that).

**13. How could you improve it?** Use convolutional layers (DCGAN), conditional GAN (choose the digit), more epochs, or evaluate with FID.

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` not recognised | Reinstall Python and tick "Add Python to PATH", or try `py -m venv venv` |
| `streamlit` not recognised | Activate the venv first (`(venv)` must show in the prompt) |
| App says "No trained model found" | Run `python train.py` first |
| MNIST download fails | Check internet connection and run `python train.py` again |
| Training feels slow | Use fewer epochs (`--epochs 10`) or a GPU with `--cuda` |
| Digits look blurry | Train for more epochs (30-50) |

## Tech stack

Python - PyTorch - torchvision - Streamlit - Matplotlib - NumPy - Pandas - Pillow
