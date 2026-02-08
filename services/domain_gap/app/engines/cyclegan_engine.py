"""
CycleGAN Engine - Unpaired Image-to-Image Translation
=======================================================
Translates synthetic images to look like real images using CycleGAN
(unpaired image-to-image translation). Implemented from scratch with
LM-CycleGAN-inspired architecture: ResNet-9blocks generator with
InstanceNorm, PatchGAN discriminator with spectral normalization,
LSGAN adversarial loss, cycle consistency, identity loss, and
LPIPS perceptual loss.

Training requires ~10GB VRAM; inference requires ~3GB.
Falls back to CPU if no GPU is available.
"""

import json
import os
import random
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from loguru import logger

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# =============================================================================
# Network Architecture
# =============================================================================


class ResidualBlock(nn.Module):
    """
    Residual block with two 3x3 convolutions, InstanceNorm, and skip connection.

    Architecture:
        x -> Conv(256,256,3) -> InstanceNorm -> ReLU -> Conv(256,256,3) -> InstanceNorm -> (+x)
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Generator(nn.Module):
    """
    ResNet-9blocks generator for CycleGAN.

    Architecture:
        - Initial:       Conv2d(3, 64, 7, padding=3) + InstanceNorm + ReLU
        - Downsampling:  2x Conv2d stride=2 (64->128->256)
        - ResNet blocks: 9x ResidualBlock(256)
        - Upsampling:    2x ConvTranspose2d (256->128->64)
        - Output:        Conv2d(64, 3, 7, padding=3) + Tanh

    Input/Output: (B, 3, H, W) in range [-1, 1]
    """

    def __init__(self, in_channels: int = 3, num_residual_blocks: int = 9) -> None:
        super().__init__()

        # Initial convolution block
        initial = [
            nn.Conv2d(in_channels, 64, kernel_size=7, padding=3, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]

        # Downsampling: 64 -> 128 -> 256
        downsampling = [
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        ]

        # Residual blocks
        residual = [ResidualBlock(256) for _ in range(num_residual_blocks)]

        # Upsampling: 256 -> 128 -> 64
        upsampling = [
            nn.ConvTranspose2d(
                256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
            ),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
            ),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]

        # Output layer
        output = [
            nn.Conv2d(64, in_channels, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(
            *initial, *downsampling, *residual, *upsampling, *output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Discriminator(nn.Module):
    """
    PatchGAN discriminator with spectral normalization.

    Architecture:
        - Layer 1: SpectralNorm(Conv2d(3, 64, 4, stride=2)) + LeakyReLU(0.2)
        - Layer 2: SpectralNorm(Conv2d(64, 128, 4, stride=2)) + InstanceNorm + LeakyReLU(0.2)
        - Layer 3: SpectralNorm(Conv2d(128, 256, 4, stride=2)) + InstanceNorm + LeakyReLU(0.2)
        - Layer 4: SpectralNorm(Conv2d(256, 512, 4, stride=1)) + InstanceNorm + LeakyReLU(0.2)
        - Output:  SpectralNorm(Conv2d(512, 1, 4, stride=1))

    Outputs a patch-level prediction map (each value = real/fake for its receptive field).
    """

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()

        layers = [
            # Layer 1: no InstanceNorm on first layer
            nn.utils.spectral_norm(
                nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)
            ),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 2
            nn.utils.spectral_norm(
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
            ),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 3
            nn.utils.spectral_norm(
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
            ),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 4
            nn.utils.spectral_norm(
                nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False)
            ),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: 1-channel patch prediction
            nn.utils.spectral_norm(
                nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
            ),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# =============================================================================
# Image Buffer (replay buffer for discriminator stability)
# =============================================================================


class ImageBuffer:
    """
    Replay buffer that stores previously generated images and returns
    a mix of new and buffered images to stabilize discriminator training.

    With 50% probability, a generated image is swapped with a random
    image from the buffer (Shrivastava et al., 2017).
    """

    def __init__(self, max_size: int = 50) -> None:
        self.max_size = max_size
        self.buffer: List[torch.Tensor] = []

    def query(self, images: torch.Tensor) -> torch.Tensor:
        """Return a batch mixing new and buffered images."""
        if self.max_size == 0:
            return images

        result = []
        for image in images:
            image = image.unsqueeze(0)  # (1, C, H, W)

            if len(self.buffer) < self.max_size:
                self.buffer.append(image.clone())
                result.append(image)
            else:
                if random.random() > 0.5:
                    # Swap with a random buffered image
                    idx = random.randint(0, self.max_size - 1)
                    old = self.buffer[idx].clone()
                    self.buffer[idx] = image.clone()
                    result.append(old)
                else:
                    result.append(image)

        return torch.cat(result, dim=0)


# =============================================================================
# Dataset
# =============================================================================


class UnpairedImageDataset(Dataset):
    """
    Dataset for unpaired image-to-image translation.

    Loads images from two directories (synthetic and real) independently.
    Applies random horizontal flip and random crop augmentation.
    """

    def __init__(
        self,
        synthetic_dir: str,
        real_dir: str,
        image_size: int = 512,
    ) -> None:
        self.image_size = image_size

        self.synthetic_paths = self._list_images(synthetic_dir)
        self.real_paths = self._list_images(real_dir)

        if not self.synthetic_paths:
            raise ValueError(f"No images found in synthetic directory: {synthetic_dir}")
        if not self.real_paths:
            raise ValueError(f"No images found in real directory: {real_dir}")

        logger.info(
            "UnpairedImageDataset: {} synthetic, {} real images",
            len(self.synthetic_paths),
            len(self.real_paths),
        )

    def __len__(self) -> int:
        return max(len(self.synthetic_paths), len(self.real_paths))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        syn_path = self.synthetic_paths[idx % len(self.synthetic_paths)]
        real_path = self.real_paths[idx % len(self.real_paths)]

        syn_img = self._load_and_preprocess(syn_path)
        real_img = self._load_and_preprocess(real_path)

        return syn_img, real_img

    def _load_and_preprocess(self, path: str) -> torch.Tensor:
        """Load image, apply augmentation, normalize to [-1, 1]."""
        img = cv2.imread(path)
        if img is None:
            raise IOError(f"Failed to read image: {path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to slightly larger than target for random cropping
        load_size = int(self.image_size * 1.12)
        img = cv2.resize(img, (load_size, load_size), interpolation=cv2.INTER_AREA)

        # Random crop to target size
        h_start = random.randint(0, load_size - self.image_size)
        w_start = random.randint(0, load_size - self.image_size)
        img = img[h_start:h_start + self.image_size, w_start:w_start + self.image_size]

        # Random horizontal flip
        if random.random() > 0.5:
            img = np.flip(img, axis=1).copy()

        # Convert to tensor and normalize to [-1, 1]
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1.0

        return tensor

    @staticmethod
    def _list_images(directory: str) -> List[str]:
        """List image files in a directory, sorted for reproducibility."""
        dir_path = Path(directory)
        if not dir_path.is_dir():
            return []
        return sorted(
            str(p) for p in dir_path.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )


# =============================================================================
# LR Scheduler (linear decay in second half of training)
# =============================================================================


class LinearDecayLR:
    """
    Linear learning rate decay starting at epoch ``decay_start``.

    LR ramps linearly from ``initial_lr`` to 0 between ``decay_start``
    and ``total_epochs``.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_epochs: int,
        initial_lr: float,
    ) -> None:
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.initial_lr = initial_lr
        self.decay_start = total_epochs // 2

    def step(self, epoch: int) -> None:
        if epoch < self.decay_start:
            lr = self.initial_lr
        else:
            decay_epochs = self.total_epochs - self.decay_start
            progress = (epoch - self.decay_start) / max(decay_epochs, 1)
            lr = self.initial_lr * (1.0 - progress)
            lr = max(lr, 0.0)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


# =============================================================================
# CycleGAN Engine
# =============================================================================


class CycleGANEngine:
    """
    CycleGAN engine for unpaired synthetic-to-real image translation.

    Handles training, inference (single and batch), and model lifecycle
    management. Models are stored in ``{models_dir}/{model_id}/`` with
    generator/discriminator weights and metadata JSON.

    Uses lazy model loading to conserve VRAM until inference is requested.
    Falls back to CPU when no GPU is available.
    """

    def __init__(
        self,
        use_gpu: bool = True,
        models_dir: str = "/shared/cyclegan_models",
    ) -> None:
        self._models_dir = models_dir
        self._use_gpu = use_gpu
        self._device = torch.device("cpu")

        if use_gpu and torch.cuda.is_available():
            self._device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(
                "CycleGANEngine initialized with GPU: {} ({:.1f}GB)",
                gpu_name,
                mem_gb,
            )
        else:
            if use_gpu:
                logger.warning(
                    "CycleGANEngine: GPU requested but not available, falling back to CPU"
                )
            logger.info("CycleGANEngine initialized on CPU")

        # Lazy-loaded generator for inference
        self._loaded_model_id: Optional[str] = None
        self._gen_s2r: Optional[Generator] = None

        # LPIPS model loaded lazily on first training run
        self._lpips_model = None

        os.makedirs(self._models_dir, exist_ok=True)

    # =========================================================================
    # Training
    # =========================================================================

    async def train(
        self,
        synthetic_dir: str,
        real_dir: str,
        model_id: str,
        epochs: int = 100,
        batch_size: int = 1,
        learning_rate: float = 2e-4,
        lambda_cycle: float = 10.0,
        lambda_identity: float = 0.5,
        lambda_lpips: float = 1.0,
        image_size: int = 512,
        save_interval: int = 10,
        progress_callback=None,
    ) -> dict:
        """
        Train a CycleGAN model from scratch on unpaired synthetic/real images.

        Args:
            synthetic_dir: Directory containing synthetic training images.
            real_dir: Directory containing real training images.
            model_id: Unique identifier for the model.
            epochs: Total training epochs.
            batch_size: Batch size (1 recommended for CycleGAN).
            learning_rate: Initial learning rate for Adam.
            lambda_cycle: Weight for cycle consistency loss.
            lambda_identity: Weight for identity loss.
            lambda_lpips: Weight for LPIPS perceptual loss.
            image_size: Training image resolution (square).
            save_interval: Save checkpoint every N epochs.
            progress_callback: Optional callable(epoch, total_epochs) for progress.

        Returns:
            Dict with training summary (epochs_trained, final losses, model_dir).

        Raises:
            ValueError: If input directories are empty or invalid.
            RuntimeError: If CUDA runs out of memory.
        """
        start_time = time.time()
        logger.info(
            "Starting CycleGAN training: model_id={}, epochs={}, image_size={}, device={}",
            model_id,
            epochs,
            image_size,
            self._device,
        )

        # Validate inputs
        if not os.path.isdir(synthetic_dir):
            raise ValueError(f"Synthetic directory not found: {synthetic_dir}")
        if not os.path.isdir(real_dir):
            raise ValueError(f"Real directory not found: {real_dir}")

        # Prepare output directory
        model_dir = os.path.join(self._models_dir, model_id)
        os.makedirs(model_dir, exist_ok=True)

        # Create dataset and dataloader
        dataset = UnpairedImageDataset(synthetic_dir, real_dir, image_size=image_size)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=(self._device.type == "cuda"),
            drop_last=True,
        )

        # Initialize networks
        gen_s2r = Generator().to(self._device)  # synthetic -> real
        gen_r2s = Generator().to(self._device)  # real -> synthetic
        disc_r = Discriminator().to(self._device)  # discriminates real images
        disc_s = Discriminator().to(self._device)  # discriminates synthetic images

        # Apply Kaiming initialization
        self._init_weights(gen_s2r)
        self._init_weights(gen_r2s)
        self._init_weights(disc_r)
        self._init_weights(disc_s)

        # Optimizers
        opt_gen = torch.optim.Adam(
            list(gen_s2r.parameters()) + list(gen_r2s.parameters()),
            lr=learning_rate,
            betas=(0.5, 0.999),
        )
        opt_disc = torch.optim.Adam(
            list(disc_r.parameters()) + list(disc_s.parameters()),
            lr=learning_rate,
            betas=(0.5, 0.999),
        )

        # LR schedulers (linear decay in second half)
        sched_gen = LinearDecayLR(opt_gen, epochs, learning_rate)
        sched_disc = LinearDecayLR(opt_disc, epochs, learning_rate)

        # Loss functions
        criterion_gan = nn.MSELoss()  # LSGAN
        criterion_cycle = nn.L1Loss()
        criterion_identity = nn.L1Loss()

        # LPIPS perceptual loss (lazy load)
        lpips_model = None
        if lambda_lpips > 0:
            lpips_model = self._get_lpips_model()

        # Image buffers for discriminator stability
        buffer_real = ImageBuffer(max_size=50)
        buffer_synth = ImageBuffer(max_size=50)

        # Training loop
        loss_history = {
            "gen_total": [],
            "disc_total": [],
            "cycle": [],
            "identity": [],
            "lpips": [],
        }

        try:
            for epoch in range(epochs):
                epoch_losses = {
                    "gen_total": 0.0,
                    "disc_total": 0.0,
                    "cycle": 0.0,
                    "identity": 0.0,
                    "lpips": 0.0,
                }
                num_batches = 0

                gen_s2r.train()
                gen_r2s.train()
                disc_r.train()
                disc_s.train()

                for batch_idx, (synth_imgs, real_imgs) in enumerate(dataloader):
                    synth_imgs = synth_imgs.to(self._device)
                    real_imgs = real_imgs.to(self._device)

                    # ---------------------------------------------------------
                    # Train Generators
                    # ---------------------------------------------------------
                    opt_gen.zero_grad()

                    # Forward translations
                    fake_real = gen_s2r(synth_imgs)  # synthetic -> fake real
                    fake_synth = gen_r2s(real_imgs)  # real -> fake synthetic

                    # Adversarial losses (LSGAN: targets = 1 for generated)
                    pred_fake_real = disc_r(fake_real)
                    pred_fake_synth = disc_s(fake_synth)
                    target_ones_r = torch.ones_like(pred_fake_real, device=self._device)
                    target_ones_s = torch.ones_like(pred_fake_synth, device=self._device)
                    loss_gan_s2r = criterion_gan(pred_fake_real, target_ones_r)
                    loss_gan_r2s = criterion_gan(pred_fake_synth, target_ones_s)
                    loss_gan = loss_gan_s2r + loss_gan_r2s

                    # Cycle consistency losses
                    recovered_synth = gen_r2s(fake_real)  # fake_real -> recovered synthetic
                    recovered_real = gen_s2r(fake_synth)  # fake_synth -> recovered real
                    loss_cycle_synth = criterion_cycle(recovered_synth, synth_imgs)
                    loss_cycle_real = criterion_cycle(recovered_real, real_imgs)
                    loss_cycle = (loss_cycle_synth + loss_cycle_real) * lambda_cycle

                    # Identity losses: gen_s2r(real) should be close to real
                    loss_ident = torch.tensor(0.0, device=self._device)
                    if lambda_identity > 0:
                        ident_real = gen_s2r(real_imgs)
                        ident_synth = gen_r2s(synth_imgs)
                        loss_ident_real = criterion_identity(ident_real, real_imgs)
                        loss_ident_synth = criterion_identity(ident_synth, synth_imgs)
                        loss_ident = (
                            (loss_ident_real + loss_ident_synth)
                            * lambda_identity
                            * lambda_cycle
                        )

                    # LPIPS perceptual loss on the translated images
                    loss_lpips = torch.tensor(0.0, device=self._device)
                    if lpips_model is not None and lambda_lpips > 0:
                        # LPIPS expects input in [-1, 1], which our images already are
                        loss_lpips_fwd = lpips_model(fake_real, synth_imgs).mean()
                        loss_lpips_bwd = lpips_model(fake_synth, real_imgs).mean()
                        loss_lpips = (loss_lpips_fwd + loss_lpips_bwd) * lambda_lpips

                    # Total generator loss
                    loss_gen = loss_gan + loss_cycle + loss_ident + loss_lpips
                    loss_gen.backward()
                    opt_gen.step()

                    # ---------------------------------------------------------
                    # Train Discriminators
                    # ---------------------------------------------------------
                    opt_disc.zero_grad()

                    # Use image buffer for stability
                    fake_real_buf = buffer_real.query(fake_real.detach())
                    fake_synth_buf = buffer_synth.query(fake_synth.detach())

                    # Discriminator for real domain
                    pred_real_r = disc_r(real_imgs)
                    pred_fake_r = disc_r(fake_real_buf)
                    target_ones_d = torch.ones_like(pred_real_r, device=self._device)
                    target_zeros_d = torch.zeros_like(pred_fake_r, device=self._device)
                    loss_disc_r = (
                        criterion_gan(pred_real_r, target_ones_d)
                        + criterion_gan(pred_fake_r, target_zeros_d)
                    ) * 0.5

                    # Discriminator for synthetic domain
                    pred_real_s = disc_s(synth_imgs)
                    pred_fake_s = disc_s(fake_synth_buf)
                    target_ones_ds = torch.ones_like(pred_real_s, device=self._device)
                    target_zeros_ds = torch.zeros_like(pred_fake_s, device=self._device)
                    loss_disc_s = (
                        criterion_gan(pred_real_s, target_ones_ds)
                        + criterion_gan(pred_fake_s, target_zeros_ds)
                    ) * 0.5

                    loss_disc = loss_disc_r + loss_disc_s
                    loss_disc.backward()
                    opt_disc.step()

                    # Accumulate losses
                    epoch_losses["gen_total"] += loss_gen.item()
                    epoch_losses["disc_total"] += loss_disc.item()
                    epoch_losses["cycle"] += loss_cycle.item()
                    epoch_losses["identity"] += loss_ident.item()
                    epoch_losses["lpips"] += loss_lpips.item()
                    num_batches += 1

                # Average epoch losses
                if num_batches > 0:
                    for key in epoch_losses:
                        epoch_losses[key] /= num_batches

                loss_history["gen_total"].append(epoch_losses["gen_total"])
                loss_history["disc_total"].append(epoch_losses["disc_total"])
                loss_history["cycle"].append(epoch_losses["cycle"])
                loss_history["identity"].append(epoch_losses["identity"])
                loss_history["lpips"].append(epoch_losses["lpips"])

                # Update learning rate
                sched_gen.step(epoch)
                sched_disc.step(epoch)

                # Log progress
                current_lr = opt_gen.param_groups[0]["lr"]
                logger.info(
                    "Epoch [{}/{}] gen={:.4f} disc={:.4f} cycle={:.4f} "
                    "ident={:.4f} lpips={:.4f} lr={:.6f}",
                    epoch + 1,
                    epochs,
                    epoch_losses["gen_total"],
                    epoch_losses["disc_total"],
                    epoch_losses["cycle"],
                    epoch_losses["identity"],
                    epoch_losses["lpips"],
                    current_lr,
                )

                # Progress callback
                if progress_callback:
                    progress_callback(epoch + 1, epochs)

                # Save checkpoint at intervals and at last epoch
                if (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs:
                    self._save_checkpoint(
                        model_dir=model_dir,
                        gen_s2r=gen_s2r,
                        gen_r2s=gen_r2s,
                        disc_r=disc_r,
                        disc_s=disc_s,
                        epoch=epoch + 1,
                    )
                    logger.info("Checkpoint saved at epoch {}", epoch + 1)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(
                    "CUDA out of memory during training. Try reducing batch_size "
                    "or image_size. Error: {}",
                    e,
                )
                # Attempt to free memory before re-raising
                self._free_gpu_memory()
            raise
        finally:
            # Always attempt to free training VRAM
            del gen_r2s, disc_r, disc_s
            if lpips_model is not None:
                del lpips_model
                self._lpips_model = None
            self._free_gpu_memory()

        # Save metadata
        training_time = time.time() - start_time
        metadata = {
            "model_id": model_id,
            "reference_set_id": os.path.basename(real_dir),
            "epochs_trained": epochs,
            "image_size": image_size,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "lambda_cycle": lambda_cycle,
            "lambda_identity": lambda_identity,
            "lambda_lpips": lambda_lpips,
            "final_losses": {
                "gen_total": loss_history["gen_total"][-1] if loss_history["gen_total"] else None,
                "disc_total": loss_history["disc_total"][-1] if loss_history["disc_total"] else None,
                "cycle": loss_history["cycle"][-1] if loss_history["cycle"] else None,
                "identity": loss_history["identity"][-1] if loss_history["identity"] else None,
                "lpips": loss_history["lpips"][-1] if loss_history["lpips"] else None,
            },
            "training_losses": {
                "gen_total": loss_history["gen_total"],
                "disc_total": loss_history["disc_total"],
            },
            "training_time_seconds": round(training_time, 2),
            "synthetic_dir": synthetic_dir,
            "real_dir": real_dir,
            "synthetic_image_count": len(
                UnpairedImageDataset._list_images(synthetic_dir)
            ),
            "real_image_count": len(
                UnpairedImageDataset._list_images(real_dir)
            ),
            "device": str(self._device),
            "created_at": datetime.now().isoformat(),
        }

        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            "CycleGAN training complete: model_id={}, epochs={}, time={:.1f}s",
            model_id,
            epochs,
            training_time,
        )

        # Cache the trained generator for immediate inference
        self._gen_s2r = gen_s2r.eval()
        self._loaded_model_id = model_id

        return {
            "model_id": model_id,
            "model_dir": model_dir,
            "epochs_trained": epochs,
            "training_time_seconds": round(training_time, 2),
            "final_losses": metadata["final_losses"],
        }

    # =========================================================================
    # Inference
    # =========================================================================

    def translate_single(
        self,
        image_path: str,
        model_id: str,
        output_path: str,
        annotations_path: Optional[str] = None,
    ) -> str:
        """
        Translate a single synthetic image to look like a real image.

        Args:
            image_path: Path to the input synthetic image.
            model_id: ID of the trained CycleGAN model to use.
            output_path: Path where the translated image will be saved.
            annotations_path: Optional path to annotations file to copy alongside.

        Returns:
            Path to the saved translated image.

        Raises:
            FileNotFoundError: If the image or model does not exist.
            RuntimeError: If translation fails.
        """
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Ensure the model is loaded
        self.load_model(model_id)

        # Load and preprocess the image
        img = cv2.imread(image_path)
        if img is None:
            raise IOError(f"Failed to read image: {image_path}")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_h, original_w = img_rgb.shape[:2]

        # Get model's training image size from metadata
        metadata = self._load_metadata(model_id)
        image_size = metadata.get("image_size", 512)

        # Resize to model's expected input size
        img_resized = cv2.resize(
            img_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA
        )

        # Convert to tensor: (H, W, C) -> (1, C, H, W), normalize to [-1, 1]
        tensor = (
            torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1.0
        )
        tensor = tensor.to(self._device)

        # Translate
        with torch.no_grad():
            translated = self._gen_s2r(tensor)

        # Convert back to image: [-1, 1] -> [0, 255]
        translated_np = (
            translated.squeeze(0).cpu().permute(1, 2, 0).numpy() * 127.5 + 127.5
        )
        translated_np = np.clip(translated_np, 0, 255).astype(np.uint8)

        # Resize back to original dimensions
        if (original_h, original_w) != (image_size, image_size):
            translated_np = cv2.resize(
                translated_np,
                (original_w, original_h),
                interpolation=cv2.INTER_LANCZOS4,
            )

        # Convert RGB to BGR for OpenCV saving
        translated_bgr = cv2.cvtColor(translated_np, cv2.COLOR_RGB2BGR)

        # Save
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        cv2.imwrite(output_path, translated_bgr)

        # Copy annotations alongside if provided
        if annotations_path and os.path.isfile(annotations_path):
            ann_ext = Path(annotations_path).suffix
            out_stem = Path(output_path).stem
            out_dir = os.path.dirname(output_path) or "."
            ann_output = os.path.join(out_dir, f"{out_stem}{ann_ext}")
            shutil.copy2(annotations_path, ann_output)

        # Free inference VRAM
        if self._device.type == "cuda":
            torch.cuda.empty_cache()

        return output_path

    def translate_batch(
        self,
        images_dir: str,
        model_id: str,
        output_dir: str,
        annotations_dir: Optional[str] = None,
        progress_callback=None,
    ) -> dict:
        """
        Translate all images in a directory from synthetic to real domain.

        Args:
            images_dir: Directory containing synthetic input images.
            model_id: ID of the trained CycleGAN model.
            output_dir: Directory where translated images will be saved.
            annotations_dir: Optional directory with annotation files to copy.
            progress_callback: Optional callable(processed, total) for progress.

        Returns:
            Dict with total_images, translated, failed counts and timing.
        """
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        # List all images
        image_files = sorted(
            p for p in Path(images_dir).iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not image_files:
            logger.warning("No images found in {}", images_dir)
            return {"total_images": 0, "translated": 0, "failed": 0}

        os.makedirs(output_dir, exist_ok=True)

        # Ensure model is loaded once before the loop
        self.load_model(model_id)

        translated_count = 0
        failed_count = 0
        start_time = time.time()

        for idx, img_path in enumerate(image_files):
            try:
                output_path = os.path.join(output_dir, img_path.name)

                # Find matching annotation file
                ann_path = None
                if annotations_dir:
                    for ann_ext in [".json", ".txt", ".xml"]:
                        candidate = Path(annotations_dir) / f"{img_path.stem}{ann_ext}"
                        if candidate.exists():
                            ann_path = str(candidate)
                            break

                self.translate_single(
                    image_path=str(img_path),
                    model_id=model_id,
                    output_path=output_path,
                    annotations_path=ann_path,
                )
                translated_count += 1

            except Exception as e:
                logger.error("Failed to translate {}: {}", img_path.name, e)
                failed_count += 1

            if progress_callback:
                progress_callback(idx + 1, len(image_files))

        processing_time = time.time() - start_time

        logger.info(
            "Batch translation complete: {} images, {} translated, {} failed in {:.1f}s",
            len(image_files),
            translated_count,
            failed_count,
            processing_time,
        )

        return {
            "total_images": len(image_files),
            "translated": translated_count,
            "failed": failed_count,
            "processing_time_seconds": round(processing_time, 2),
        }

    # =========================================================================
    # Model Management
    # =========================================================================

    def list_models(self) -> List[dict]:
        """
        List all trained CycleGAN models with their metadata.

        Returns:
            List of dicts with model_id, reference_set_id, epochs_trained,
            image_size, model_dir, and created_at.
        """
        models = []
        models_path = Path(self._models_dir)

        if not models_path.is_dir():
            return models

        for entry in sorted(models_path.iterdir()):
            if not entry.is_dir():
                continue

            metadata_path = entry / "metadata.json"
            gen_path = entry / "gen_s2r.pth"

            # Only list directories that have both metadata and generator weights
            if not metadata_path.is_file() or not gen_path.is_file():
                continue

            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

                models.append({
                    "model_id": metadata.get("model_id", entry.name),
                    "reference_set_id": metadata.get("reference_set_id", "unknown"),
                    "epochs_trained": metadata.get("epochs_trained", 0),
                    "image_size": metadata.get("image_size", 512),
                    "final_losses": metadata.get("final_losses"),
                    "model_dir": str(entry),
                    "created_at": metadata.get("created_at"),
                    "training_time_seconds": metadata.get("training_time_seconds"),
                })
            except Exception as e:
                logger.warning("Failed to read metadata for {}: {}", entry.name, e)
                continue

        logger.debug("Found {} CycleGAN models", len(models))
        return models

    def delete_model(self, model_id: str) -> bool:
        """
        Delete a trained CycleGAN model and all its files.

        Args:
            model_id: ID of the model to delete.

        Returns:
            True if deleted, False if model was not found.
        """
        model_dir = os.path.join(self._models_dir, model_id)

        if not os.path.isdir(model_dir):
            logger.warning("Model not found for deletion: {}", model_id)
            return False

        # Unload from memory if this model is currently loaded
        if self._loaded_model_id == model_id:
            self._gen_s2r = None
            self._loaded_model_id = None
            self._free_gpu_memory()
            logger.info("Unloaded model {} from memory before deletion", model_id)

        shutil.rmtree(model_dir)
        logger.info("Deleted CycleGAN model: {}", model_id)
        return True

    def load_model(self, model_id: str) -> None:
        """
        Load a trained CycleGAN generator (synthetic->real) into memory.

        This is a lazy operation: if the model is already loaded, this is a no-op.

        Args:
            model_id: ID of the model to load.

        Raises:
            FileNotFoundError: If model directory or weights file does not exist.
        """
        # Already loaded
        if self._loaded_model_id == model_id and self._gen_s2r is not None:
            return

        model_dir = os.path.join(self._models_dir, model_id)
        gen_path = os.path.join(model_dir, "gen_s2r.pth")

        if not os.path.isfile(gen_path):
            raise FileNotFoundError(
                f"Generator weights not found: {gen_path}"
            )

        logger.info("Loading CycleGAN model: {} on {}", model_id, self._device)
        start_time = time.time()

        # Free previous model from memory
        if self._gen_s2r is not None:
            del self._gen_s2r
            self._gen_s2r = None
            self._free_gpu_memory()

        try:
            gen = Generator()
            state_dict = torch.load(
                gen_path, map_location=self._device, weights_only=True
            )
            gen.load_state_dict(state_dict)
            gen.to(self._device)
            gen.eval()

            self._gen_s2r = gen
            self._loaded_model_id = model_id

            load_time = time.time() - start_time
            logger.info(
                "CycleGAN model {} loaded in {:.2f}s on {}",
                model_id,
                load_time,
                self._device,
            )

        except Exception as e:
            logger.error("Failed to load CycleGAN model {}: {}", model_id, e)
            self._gen_s2r = None
            self._loaded_model_id = None
            self._free_gpu_memory()
            raise

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _save_checkpoint(
        self,
        model_dir: str,
        gen_s2r: Generator,
        gen_r2s: Generator,
        disc_r: Discriminator,
        disc_s: Discriminator,
        epoch: int,
    ) -> None:
        """Save all network weights to the model directory."""
        torch.save(gen_s2r.state_dict(), os.path.join(model_dir, "gen_s2r.pth"))
        torch.save(gen_r2s.state_dict(), os.path.join(model_dir, "gen_r2s.pth"))
        torch.save(disc_r.state_dict(), os.path.join(model_dir, "disc_r.pth"))
        torch.save(disc_s.state_dict(), os.path.join(model_dir, "disc_s.pth"))
        logger.debug("Checkpoint saved to {} at epoch {}", model_dir, epoch)

    def _load_metadata(self, model_id: str) -> dict:
        """Load model metadata from JSON file."""
        metadata_path = os.path.join(self._models_dir, model_id, "metadata.json")
        if not os.path.isfile(metadata_path):
            logger.warning("Metadata not found for model {}, using defaults", model_id)
            return {}

        with open(metadata_path, "r") as f:
            return json.load(f)

    def _get_lpips_model(self):
        """Lazily load the LPIPS perceptual loss model."""
        if self._lpips_model is not None:
            return self._lpips_model

        try:
            import lpips

            logger.info("Loading LPIPS model (VGG backbone)...")
            self._lpips_model = lpips.LPIPS(net="vgg").to(self._device)
            self._lpips_model.eval()
            # Freeze LPIPS weights - it is only used as a loss function
            for param in self._lpips_model.parameters():
                param.requires_grad = False
            logger.info("LPIPS model loaded on {}", self._device)
            return self._lpips_model

        except ImportError:
            logger.warning(
                "lpips package not installed. LPIPS perceptual loss will be disabled. "
                "Install with: pip install lpips"
            )
            return None
        except Exception as e:
            logger.error("Failed to load LPIPS model: {}", e)
            return None

    @staticmethod
    def _init_weights(net: nn.Module) -> None:
        """
        Apply Kaiming normal initialization to Conv2d and ConvTranspose2d layers.
        InstanceNorm2d parameters (if any affine) are initialized to N(1, 0.02) / 0.
        """
        for m in net.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    nn.init.normal_(m.weight, 1.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _free_gpu_memory(self) -> None:
        """Clear CUDA cache to free unused GPU memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
