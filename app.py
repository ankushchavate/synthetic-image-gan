"""
app.py - Streamlit user interface
=================================
Synthetic Image Generation Using Deep Learning (GAN on MNIST)

Run with:   streamlit run app.py

Tabs:
  1. Learn          - beginner-friendly explanation of GANs
  2. Generate       - create new digits, view them, download them
  3. Real vs Fake   - compare real MNIST digits with generated ones
  4. Training Results - loss plot and how the digits improved per epoch
  5. Train          - (optional) train the GAN directly from the browser
"""

import io
import zipfile
from datetime import datetime

import matplotlib

matplotlib.use("Agg")  # non-interactive backend, safe for Streamlit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from torchvision import datasets

from model import (
    CHECKPOINT_PATH,
    DATA_DIR,
    GENERATED_DIR,
    GENERATOR_PATH,
    SAMPLES_DIR,
    get_device,
    load_generator,
)
from train import DEFAULT_CONFIG, load_history, train

st.set_page_config(page_title="Synthetic Image Generation (GAN)", page_icon="🖼️", layout="wide")


# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------
def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """Generator output in [-1, 1], shape (N,1,28,28) -> uint8 array (N,28,28)."""
    t = (t.clamp(-1, 1) + 1) / 2 * 255
    return t.round().byte().squeeze(1).cpu().numpy()


def make_grid_image(images: np.ndarray, ncols: int, scale: int = 4, pad: int = 2) -> Image.Image:
    """Arrange (N,28,28) uint8 images into one grid picture."""
    n, h, w = images.shape
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    canvas = np.full((nrows * (h + pad) + pad, ncols * (w + pad) + pad), 60, dtype=np.uint8)
    for idx in range(n):
        r, c = divmod(idx, ncols)
        y, x = pad + r * (h + pad), pad + c * (w + pad)
        canvas[y:y + h, x:x + w] = images[idx]
    img = Image.fromarray(canvas)
    return img.resize((img.width * scale, img.height * scale), Image.NEAREST)


def image_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_zip(images: np.ndarray, scale: int = 8) -> bytes:
    """Pack every image as an individual PNG inside one ZIP file."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, arr in enumerate(images, start=1):
            im = Image.fromarray(arr).resize((28 * scale, 28 * scale), Image.NEAREST)
            zf.writestr(f"digit_{i:03d}.png", image_to_png_bytes(im))
    return buf.getvalue()


@st.cache_resource(show_spinner=False)
def cached_generator(path_str: str, mtime: float, use_cuda: bool):
    """Load the generator once. 'mtime' changes after re-training, so the
    cache refreshes automatically when a new model file is saved."""
    device = get_device(use_cuda)
    generator, info = load_generator(path_str, device)
    return generator, info, device


def generate_images(generator, latent_dim, device, count, seed):
    """Create `count` images from random noise (reproducible with a seed)."""
    g = torch.Generator().manual_seed(int(seed))
    z = torch.randn(count, latent_dim, generator=g).to(device)
    with torch.no_grad():
        return tensor_to_uint8(generator(z))


@st.cache_resource(show_spinner=False)
def cached_real_dataset():
    """MNIST test images (uint8 array). Returns None if not downloaded yet."""
    try:
        ds = datasets.MNIST(root=str(DATA_DIR), train=False, download=False)
        return ds.data.numpy()
    except Exception:
        return None


def get_model_or_warn(use_cuda: bool):
    """Return (generator, info, device), or None after showing a friendly message.
    (We never call st.stop() so the other tabs, e.g. Train, keep working.)"""
    if not GENERATOR_PATH.exists():
        st.warning(
            "⚠️ **No trained model found.** Train the GAN first:\n\n"
            "- In a terminal run `python train.py`, **or**\n"
            "- Open the **Train** tab in this app.\n\n"
            "Then refresh this page."
        )
        return None
    try:
        return cached_generator(str(GENERATOR_PATH), GENERATOR_PATH.stat().st_mtime, use_cuda)
    except (FileNotFoundError, RuntimeError) as exc:
        st.error(f"❌ {exc}")
        return None


def plot_losses(history):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(history["epoch"], history["g_loss"], label="Generator loss", color="tab:blue")
    ax.plot(history["epoch"], history["d_loss"], label="Discriminator loss", color="tab:red")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average loss")
    ax.set_title("GAN training losses")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")
cuda_available = torch.cuda.is_available()
use_cuda = st.sidebar.checkbox(
    "Use GPU (CUDA)", value=False, disabled=not cuda_available,
    help="CPU is used by default. Enable only if you have an NVIDIA GPU with CUDA PyTorch.",
)
if not cuda_available:
    st.sidebar.caption("CUDA GPU not detected - running on CPU.")
st.sidebar.markdown("---")
if GENERATOR_PATH.exists():
    st.sidebar.success("Trained model found ✅")
else:
    st.sidebar.error("No trained model yet ❌")

st.title("🖼️ Synthetic Image Generation Using Deep Learning")
st.caption("A simple GAN (PyTorch) trained on the MNIST handwritten digits dataset.")

tab_learn, tab_gen, tab_cmp, tab_res, tab_train = st.tabs(
    ["📘 Learn", "✨ Generate", "🔍 Real vs Generated", "📈 Training Results", "🏋️ Train"]
)

# --------------------------------------------------------------------------
# Tab 1: Learn
# --------------------------------------------------------------------------
with tab_learn:
    st.header("What is synthetic image generation?")
    st.write(
        "**Synthetic images** are pictures created by a computer program instead of a camera "
        "or a human. They look like real data but are brand new. They are useful when real data "
        "is scarce, private (e.g. medical images), or expensive to collect, and for training "
        "other AI models."
    )

    st.header("What is a GAN?")
    st.write(
        "A **Generative Adversarial Network (GAN)** has two neural networks that compete:"
    )
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🎨 Generator (the forger)")
        st.write(
            "Takes a vector of **random noise** and turns it into an image. At the start the "
            "output looks like static. Its goal is to produce images so good that the "
            "Discriminator thinks they are real."
        )
    with c2:
        st.subheader("🕵️ Discriminator (the detective)")
        st.write(
            "Looks at an image and says how likely it is to be **real** (from the dataset) or "
            "**fake** (from the Generator). Its goal is to catch the forger."
        )
    st.info(
        "**Analogy:** a counterfeiter (Generator) keeps printing fake banknotes, and a police "
        "officer (Discriminator) keeps learning to spot them. Each gets better because of the "
        "other, until the fakes look very real."
    )

    st.header("How training works (step by step)")
    st.markdown(
        """
1. Take a batch of **real** MNIST digits.
2. Make a batch of **fake** digits by giving random noise to the Generator.
3. **Train the Discriminator** to give high scores to real and low scores to fake images.
4. **Train the Generator** so the Discriminator gives its fakes a high score.
5. Repeat for many **epochs** (an epoch = one pass over all 60,000 training images).
"""
    )

    st.header("Project details")
    st.markdown(
        """
| Item | Value |
|---|---|
| Dataset | MNIST (60,000 handwritten digits, 28x28 grey images) |
| Framework | PyTorch |
| Generator | 100 noise values -> 256 -> 512 -> 1024 -> 784 pixels (Tanh) |
| Discriminator | 784 pixels -> 1024 -> 512 -> 256 -> 1 score |
| Loss | Binary cross-entropy (BCEWithLogitsLoss) |
| Optimizer | Adam (lr = 0.0002, beta1 = 0.5) |
"""
    )
    st.caption(
        "Note: the training digit labels (0-9) are never used. The GAN learns the look of "
        "digits without being told which digit is which (unsupervised learning)."
    )

# --------------------------------------------------------------------------
# Tab 2: Generate
# --------------------------------------------------------------------------
def render_generate_tab():
    st.header("Generate new digit images")
    loaded = get_model_or_warn(use_cuda)
    if loaded is None:
        return
    generator, info, device = loaded
    st.caption(f"Model trained for **{info['epochs_trained']}** epochs • running on **{device}**")

    col_a, col_b, col_c = st.columns(3)
    count = col_a.slider("Number of images", 1, 100, 16)
    ncols = col_b.slider("Grid columns", 1, 10, 8)
    seed = col_c.number_input("Random seed", min_value=0, max_value=2**31 - 1, value=42, step=1,
                              help="Same seed = same images. Change it for new digits.")
    save_to_disk = st.checkbox("Also save these images into the generated_images/ folder", value=False)

    b1, b2 = st.columns([1, 1])
    if b1.button("🎲 Generate", type="primary"):
        try:
            imgs = generate_images(generator, info["latent_dim"], device, count, seed)
            st.session_state["gen_images"] = imgs
            st.session_state["gen_ncols"] = ncols
            st.session_state["gen_saved"] = None
            if save_to_disk:
                folder = GENERATED_DIR / f"gen_{datetime.now():%Y%m%d_%H%M%S}"
                folder.mkdir(parents=True, exist_ok=True)
                for i, arr in enumerate(imgs, start=1):
                    Image.fromarray(arr).resize((224, 224), Image.NEAREST).save(
                        folder / f"digit_{i:03d}.png")
                make_grid_image(imgs, ncols).save(folder / "grid.png")
                st.session_state["gen_saved"] = str(folder)
        except Exception as exc:
            st.error(f"Image generation failed: {exc}")
    if b2.button("🎲 Generate with a random new seed"):
        new_seed = int(np.random.randint(0, 2**31 - 1))
        try:
            imgs = generate_images(generator, info["latent_dim"], device, count, new_seed)
            st.session_state["gen_images"] = imgs
            st.session_state["gen_ncols"] = ncols
            st.session_state["gen_saved"] = None
            st.info(f"Used random seed: {new_seed}")
        except Exception as exc:
            st.error(f"Image generation failed: {exc}")

    imgs = st.session_state.get("gen_images")
    if imgs is not None:
        grid = make_grid_image(imgs, st.session_state.get("gen_ncols", ncols))
        st.image(grid, caption=f"{len(imgs)} generated images (28x28, enlarged for display)")
        if st.session_state.get("gen_saved"):
            st.success(f"Saved to: {st.session_state['gen_saved']}")

        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Download grid (PNG)", image_to_png_bytes(grid),
                           file_name="generated_grid.png", mime="image/png")
        d2.download_button("⬇️ Download all images (ZIP)", make_zip(imgs),
                           file_name="generated_digits.zip", mime="application/zip")

        with st.expander("Download a single image"):
            idx = st.number_input("Image number", 1, len(imgs), 1, key=f"single_idx_{len(imgs)}")
            single = Image.fromarray(imgs[idx - 1]).resize((224, 224), Image.NEAREST)
            st.image(single, width=160)
            st.download_button("⬇️ Download this image", image_to_png_bytes(single),
                               file_name=f"digit_{idx:03d}.png", mime="image/png")

with tab_gen:
    render_generate_tab()

# --------------------------------------------------------------------------
# Tab 3: Real vs Generated
# --------------------------------------------------------------------------
def render_compare_tab():
    st.header("Real vs generated digits")
    loaded = get_model_or_warn(use_cuda)
    if loaded is None:
        return
    generator, info, device = loaded

    real_data = cached_real_dataset()
    if real_data is None:
        st.info("The MNIST dataset is not on this computer yet. It is normally downloaded by "
                "`python train.py`. You can also download it now (about 11 MB, free).")
        if st.button("Download MNIST now"):
            try:
                with st.spinner("Downloading MNIST..."):
                    datasets.MNIST(root=str(DATA_DIR), train=False, download=True)
                cached_real_dataset.clear()
                st.rerun()
            except Exception as exc:
                st.error(f"Download failed (check your internet connection): {exc}")
    else:
        n = st.slider("Images per side", 4, 32, 16, key="cmp_n")
        cmp_seed = st.number_input("Seed", 0, 2**31 - 1, 7, key="cmp_seed")
        rng = np.random.default_rng(int(cmp_seed))
        real_imgs = real_data[rng.choice(len(real_data), size=n, replace=False)]
        try:
            fake_imgs = generate_images(generator, info["latent_dim"], device, n, cmp_seed)
        except Exception as exc:
            st.error(f"Image generation failed: {exc}")
            return

        left, right = st.columns(2)
        cols = 8 if n >= 8 else n
        with left:
            st.subheader("Real (MNIST)")
            st.image(make_grid_image(real_imgs, cols))
        with right:
            st.subheader("Generated (GAN)")
            st.image(make_grid_image(fake_imgs, cols))
        st.caption("Can you tell which side is fake? The better the GAN is trained, the harder it is.")

with tab_cmp:
    render_compare_tab()

# --------------------------------------------------------------------------
# Tab 4: Training Results
# --------------------------------------------------------------------------
with tab_res:
    st.header("Training results")
    history = load_history()
    if not history or not history.get("epoch"):
        st.warning("No training history found. Run `python train.py` or use the Train tab.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Epochs trained", history["epoch"][-1])
        m2.metric("Final generator loss", f"{history['g_loss'][-1]:.3f}")
        m3.metric("Final discriminator loss", f"{history['d_loss'][-1]:.3f}")

        st.pyplot(plot_losses(history))
        st.caption(
            "In a GAN the two losses do not simply go down. They push against each other. "
            "Healthy training shows both losses staying in a stable range (neither goes to 0)."
        )

        samples = sorted(SAMPLES_DIR.glob("epoch_*.png")) if SAMPLES_DIR.exists() else []
        if samples:
            st.subheader("How the generated digits improved")
            if len(samples) > 1:
                pick = st.select_slider(
                    "Choose epoch", options=list(range(1, len(samples) + 1)), value=len(samples))
            else:
                pick = 1
            st.image(str(samples[pick - 1]), caption=f"Samples after epoch {pick}", width=450)

            if len(samples) >= 3:
                st.write("**Progress snapshot (first / middle / last epoch):**")
                idxs = [0, len(samples) // 2, len(samples) - 1]
                cols = st.columns(3)
                for col, i in zip(cols, idxs):
                    col.image(str(samples[i]), caption=f"Epoch {i + 1}")

# --------------------------------------------------------------------------
# Tab 5: Train
# --------------------------------------------------------------------------
with tab_train:
    st.header("Train the GAN from the app")
    st.write(
        "You can also train from the terminal with `python train.py` (recommended for "
        "longer runs). The first run downloads MNIST automatically (internet needed once)."
    )
    if CHECKPOINT_PATH.exists():
        st.info("A checkpoint exists. Tick **Resume** to continue, or leave it off to start over.")

    with st.form("train_form"):
        t1, t2, t3 = st.columns(3)
        epochs = t1.number_input("Epochs (total)", 1, 200, 10)
        batch_size = t2.selectbox("Batch size", [64, 128, 256], index=1)
        lr = t3.number_input("Learning rate", 0.00001, 0.01, 0.0002, step=0.0001, format="%.5f")
        resume = st.checkbox("Resume from last checkpoint", value=False)
        st.caption("On CPU, expect roughly 5-20 seconds per epoch depending on your computer.")
        submitted = st.form_submit_button("🚀 Start training", type="primary")

    if submitted:
        bar = st.progress(0.0)
        status = st.empty()
        chart_slot = st.empty()

        def on_progress(info_):
            bar.progress(float(min(max(info_["progress"], 0.0), 1.0)))
            status.write(f"**{info_['message']}**")
            hist = info_["history"]
            if hist and hist.get("epoch"):
                chart_slot.line_chart(
                    pd.DataFrame({"Generator": hist["g_loss"], "Discriminator": hist["d_loss"]},
                                 index=hist["epoch"]))

        try:
            cfg = {**DEFAULT_CONFIG, "epochs": int(epochs), "batch_size": int(batch_size),
                   "lr": float(lr), "resume": bool(resume), "use_cuda": bool(use_cuda)}
            with st.spinner("Training... please keep this page open."):
                train(cfg, progress_callback=on_progress)
            bar.progress(1.0)
            st.success("✅ Training finished! Open the other tabs to see the results.")
        except (ValueError, RuntimeError) as exc:
            st.error(f"❌ {exc}")
        except Exception as exc:
            st.error(f"Unexpected error during training: {exc}")
