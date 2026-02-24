"""
Diffusion Refinement Engine
============================
Diffusion-based post-processing engine for closing the synthetic-to-real domain
gap. Uses Stable Diffusion 1.5 as backbone with three interchangeable conditioning
strategies:

- ControlNet (depth-guided img2img): Injects geometric structure via a depth map
  so the diffusion process cannot deviate far from the original layout.
  Requires ~4.5 GB VRAM.

- IP-Adapter (zero-shot style transfer): Conditions generation on a reference
  image embedding without any fine-tuning.  Shares the ControlNet pipeline.

- LoRA (fine-tuned style): Lightweight adapter weights (~50 MB) trained on the
  real reference set via DreamBooth-style LoRA.  After training, inference cost
  is identical to vanilla img2img.

Annotation preservation is validated automatically using Canny-edge IoU, SSIM,
and ORB keypoint displacement between the original synthetic image and the
refined output.  Strength is hard-capped at 0.7 to prevent the model from
drifting too far from the original geometry.

GPU usage: ~4-6 GB VRAM for inference.  CPU fallback is supported but slow.
"""

import json
import os
import random
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
import torch
from loguru import logger

# Supported image extensions (consistent with other engines)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# Hard cap on denoising strength to preserve annotation structure
MAX_STRENGTH = 0.7


# =============================================================================
# DiffusionRefinementEngine
# =============================================================================


class DiffusionRefinementEngine:
    """Diffusion-based refinement engine for synthetic dataset post-processing.

    All diffusion model imports are deferred until first use so the service
    starts even when the ``diffusers`` package is not installed.  When
    diffusers is absent, every public method returns an error dict instead
    of raising.

    Attributes:
        _pipe: Loaded StableDiffusion pipeline (None until first use).
        _controlnet: Loaded ControlNet model (None until first use).
        _ip_adapter_loaded: Whether IP-Adapter weights are mounted on _pipe.
        _loaded_lora_id: model_id of the currently mounted LoRA, or None.
        _device: torch.device used for inference.
        _models_dir: Directory for downloaded/cached diffusion backbone weights.
        _lora_dir: Directory for trained LoRA adapter weights.
    """

    def __init__(
        self,
        use_gpu: bool = True,
        models_dir: str = "/shared/diffusion_models",
        lora_dir: str = "/shared/lora_weights",
    ) -> None:
        """Initialise the engine and configure the compute device.

        Args:
            use_gpu: Use CUDA if available; falls back to CPU silently.
            models_dir: Root directory for cached SD / ControlNet weights.
            lora_dir: Root directory for trained LoRA adapter weights.
        """
        self._device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )

        # Lazy model references
        self._pipe = None
        self._controlnet = None
        self._ip_adapter_loaded: bool = False
        self._loaded_lora_id: Optional[str] = None

        self._models_dir = Path(models_dir)
        self._lora_dir = Path(lora_dir)
        self._models_dir.mkdir(parents=True, exist_ok=True)
        self._lora_dir.mkdir(parents=True, exist_ok=True)

        if self._device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            logger.info(
                "DiffusionRefinementEngine initialized with GPU: {} ({:.1f} GB)",
                gpu_name,
                mem_gb,
            )
        else:
            if use_gpu:
                logger.warning(
                    "DiffusionRefinementEngine: GPU requested but not available, "
                    "falling back to CPU"
                )
            logger.info("DiffusionRefinementEngine initialized on CPU")

    # =========================================================================
    # VRAM Check
    # =========================================================================

    def _check_vram(self, min_gb: float = 6.0) -> bool:
        """Return True when at least ``min_gb`` GB of free VRAM is available.

        Always returns False on CPU devices.

        Args:
            min_gb: Minimum free VRAM in gigabytes required.

        Returns:
            bool indicating whether the VRAM threshold is met.
        """
        if self._device.type != "cuda":
            return False
        free_mem = torch.cuda.mem_get_info()[0] / (1024 ** 3)
        logger.debug("Free VRAM: {:.2f} GB (required: {:.1f} GB)", free_mem, min_gb)
        return free_mem >= min_gb

    # =========================================================================
    # Lazy Pipeline Loading
    # =========================================================================

    def _load_controlnet_pipeline(self) -> None:
        """Load SD 1.5 + ControlNet-depth pipeline (~4.5 GB VRAM).

        Uses fp16 precision on CUDA.  Enables cpu_offload when free VRAM is
        below 8 GB to trade speed for memory.  Always enables attention slicing.
        Sets ``self._pipe`` and ``self._controlnet`` when successful.

        Raises:
            RuntimeError: When diffusers is not installed.
        """
        if self._pipe is not None:
            return

        logger.info("Loading ControlNet-depth + SD 1.5 pipeline...")
        start = time.time()

        try:
            from diffusers import (
                ControlNetModel,
                StableDiffusionControlNetImg2ImgPipeline,
            )
        except ImportError as exc:
            raise RuntimeError(
                "diffusers package is not installed. "
                "Install with: pip install diffusers transformers accelerate"
            ) from exc

        try:
            dtype = torch.float16 if self._device.type == "cuda" else torch.float32

            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11f1p_sd15_depth",
                torch_dtype=dtype,
                cache_dir=str(self._models_dir),
            )

            pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=controlnet,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
                cache_dir=str(self._models_dir),
            )

            # Memory optimizations
            pipe.enable_attention_slicing()

            if self._device.type == "cuda":
                free_gb = torch.cuda.mem_get_info()[0] / (1024 ** 3)
                if free_gb < 8.0:
                    logger.info(
                        "Free VRAM {:.1f} GB < 8 GB — enabling model CPU offload",
                        free_gb,
                    )
                    pipe.enable_model_cpu_offload()
                else:
                    pipe.to(self._device)
            else:
                pipe.to(self._device)

            self._controlnet = controlnet
            self._pipe = pipe

            elapsed = time.time() - start
            logger.info(
                "ControlNet pipeline loaded in {:.2f}s on {}", elapsed, self._device
            )

        except Exception as exc:
            logger.error("Failed to load ControlNet pipeline: {}", exc)
            self._pipe = None
            self._controlnet = None
            raise

    def _load_ip_adapter(self) -> None:
        """Mount IP-Adapter weights onto the existing pipeline.

        Requires ``_load_controlnet_pipeline()`` to have been called first.
        Sets ``self._ip_adapter_loaded = True`` on success.
        """
        if self._ip_adapter_loaded:
            return

        if self._pipe is None:
            self._load_controlnet_pipeline()

        logger.info("Loading IP-Adapter weights...")
        try:
            self._pipe.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="models",
                weight_name="ip-adapter_sd15.bin",
                cache_dir=str(self._models_dir),
            )
            self._ip_adapter_loaded = True
            logger.info("IP-Adapter loaded successfully")
        except Exception as exc:
            logger.error("Failed to load IP-Adapter: {}", exc)
            raise

    def _load_lora(self, lora_model_id: str) -> None:
        """Mount LoRA adapter weights onto the existing pipeline.

        Args:
            lora_model_id: Identifier matching a subdirectory in
                ``self._lora_dir`` that contains the adapter weights.
        """
        if self._pipe is None:
            self._load_controlnet_pipeline()

        lora_path = self._lora_dir / lora_model_id
        if not lora_path.is_dir():
            raise FileNotFoundError(
                f"LoRA model directory not found: {lora_path}"
            )

        logger.info("Loading LoRA weights: {}", lora_model_id)
        try:
            self._pipe.load_lora_weights(str(lora_path))
            self._loaded_lora_id = lora_model_id
            logger.info("LoRA {} loaded successfully", lora_model_id)
        except Exception as exc:
            logger.error("Failed to load LoRA {}: {}", lora_model_id, exc)
            raise

    def _unload_lora(self) -> None:
        """Unload the currently mounted LoRA weights from the pipeline."""
        if self._pipe is None or self._loaded_lora_id is None:
            return
        try:
            self._pipe.unload_lora_weights()
            logger.info("LoRA {} unloaded", self._loaded_lora_id)
        except Exception as exc:
            logger.warning("Could not unload LoRA cleanly: {}", exc)
        finally:
            self._loaded_lora_id = None

    # =========================================================================
    # Single Image Refinement (Public API)
    # =========================================================================

    @torch.no_grad()
    def refine_single(
        self,
        image_path: str,
        output_path: str,
        config: dict,
        depth_map_path: Optional[str] = None,
        reference_dir: Optional[str] = None,
        annotations_path: Optional[str] = None,
    ) -> dict:
        """Refine a single synthetic image using the configured diffusion method.

        Strength is capped at MAX_STRENGTH (0.7) regardless of the value
        supplied in ``config`` so that annotation geometry is always preserved.

        Args:
            image_path: Path to the synthetic source image.
            output_path: Destination path for the refined image.
            config: Dict representation of DiffusionRefinementConfig fields.
            depth_map_path: Optional pre-computed depth map for ControlNet.
            reference_dir: Directory with real reference images (used by
                ip_adapter for reference image selection).
            annotations_path: If provided, the annotation file is copied
                alongside the output image (preserving extension).

        Returns:
            Dict with keys:
                ``success`` (bool),
                ``output_path`` (str),
                ``method`` (str),
                ``preservation_metrics`` (dict or None),
                ``processing_time_ms`` (float),
                ``error`` (str or None).
        """
        start_time = time.time()
        method = config.get("method", "controlnet")

        def _error(msg: str) -> dict:
            elapsed = (time.time() - start_time) * 1000
            logger.error("refine_single failed [{}]: {}", method, msg)
            return {
                "success": False,
                "output_path": output_path,
                "method": method,
                "preservation_metrics": None,
                "processing_time_ms": elapsed,
                "error": msg,
            }

        # --- Validate inputs --------------------------------------------------
        if not os.path.isfile(image_path):
            return _error(f"Input image not found: {image_path}")

        # Cap strength
        config = dict(config)
        config["strength"] = min(float(config.get("strength", 0.4)), MAX_STRENGTH)

        # --- Load source image ------------------------------------------------
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return _error(f"Could not read image: {image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        from PIL import Image as PILImage
        original_pil = PILImage.fromarray(img_rgb)

        # --- Load depth map ---------------------------------------------------
        depth_pil: Optional[object] = None
        if depth_map_path and os.path.isfile(depth_map_path):
            depth_pil = self._load_depth_pil(depth_map_path, original_pil.size)

        # --- Dispatch to method -----------------------------------------------
        try:
            if method == "controlnet":
                refined_rgb = self._refine_controlnet(original_pil, depth_pil, config)
            elif method == "ip_adapter":
                reference_image = None
                if reference_dir and os.path.isdir(reference_dir):
                    ref_path = self._pick_random_reference(reference_dir)
                    ref_bgr = cv2.imread(ref_path)
                    if ref_bgr is not None:
                        ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
                        reference_image = PILImage.fromarray(ref_rgb)
                if reference_image is None:
                    logger.warning(
                        "ip_adapter: no valid reference image found in {}, "
                        "falling back to controlnet method",
                        reference_dir,
                    )
                    refined_rgb = self._refine_controlnet(original_pil, depth_pil, config)
                else:
                    refined_rgb = self._refine_ip_adapter(
                        original_pil, reference_image, depth_pil, config
                    )
            elif method == "lora":
                refined_rgb = self._refine_lora(original_pil, depth_pil, config)
            else:
                return _error(f"Unknown diffusion method: {method}")
        except Exception as exc:
            return _error(str(exc))

        # --- Save refined image -----------------------------------------------
        refined_bgr = cv2.cvtColor(refined_rgb, cv2.COLOR_RGB2BGR)
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        if not cv2.imwrite(output_path, refined_bgr):
            return _error(f"Failed to write output image: {output_path}")

        # --- Annotation preservation metrics ----------------------------------
        preservation_metrics: Optional[dict] = None
        if config.get("validate_annotations", True):
            threshold = float(config.get("annotation_threshold", 0.7))
            preservation_metrics = self.validate_annotations(
                img_rgb, refined_rgb, threshold=threshold
            )

        # --- Copy annotations alongside ---------------------------------------
        if annotations_path and os.path.isfile(annotations_path):
            ann_ext = Path(annotations_path).suffix
            ann_out = str(Path(output_path).with_suffix(ann_ext))
            shutil.copy2(annotations_path, ann_out)

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            "refine_single complete: {} -> {} ({:.0f} ms, method={})",
            image_path,
            output_path,
            elapsed_ms,
            method,
        )

        return {
            "success": True,
            "output_path": output_path,
            "method": method,
            "preservation_metrics": preservation_metrics,
            "processing_time_ms": elapsed_ms,
            "error": None,
        }

    # =========================================================================
    # Internal Refinement Methods
    # =========================================================================

    @torch.no_grad()
    def _refine_controlnet(
        self,
        image,
        depth_map,
        config: dict,
    ) -> np.ndarray:
        """ControlNet depth-conditioned img2img refinement.

        If no depth map is supplied, it is estimated on the fly using
        ``controlnet_aux.MidasDetector`` (falls back gracefully when not
        installed by using a uniform grey map instead).

        Args:
            image: PIL.Image RGB source image.
            depth_map: PIL.Image grayscale depth map, or None.
            config: DiffusionRefinementConfig fields as dict.

        Returns:
            Refined image as RGB numpy uint8 array.
        """
        self._load_controlnet_pipeline()

        use_lcm: bool = config.get("use_lcm", False)
        if use_lcm:
            self._apply_lcm_scheduler()

        # Build or use depth map
        if depth_map is None:
            depth_map = self._estimate_depth(image)

        # Ensure size consistency
        if depth_map.size != image.size:
            depth_map = depth_map.resize(image.size)

        seed = config.get("seed")
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self._device).manual_seed(int(seed))

        num_steps = 4 if use_lcm else int(config.get("num_inference_steps", 30))

        output = self._pipe(
            prompt=config.get("prompt", "photorealistic, high quality"),
            negative_prompt=config.get(
                "negative_prompt", "blurry, distorted, artifacts, watermark"
            ),
            image=image,
            control_image=depth_map,
            strength=float(config.get("strength", 0.4)),
            controlnet_conditioning_scale=float(
                config.get("controlnet_conditioning_scale", 0.8)
            ),
            guidance_scale=float(config.get("guidance_scale", 7.5)),
            num_inference_steps=num_steps,
            generator=generator,
        )

        return np.array(output.images[0])

    @torch.no_grad()
    def _refine_ip_adapter(
        self,
        image,
        reference_image,
        depth_map,
        config: dict,
    ) -> np.ndarray:
        """IP-Adapter zero-shot style conditioning.

        Loads IP-Adapter weights onto the pipeline on first call.

        Args:
            image: PIL.Image RGB source image.
            reference_image: PIL.Image reference style image.
            depth_map: PIL.Image depth map, or None.
            config: DiffusionRefinementConfig fields as dict.

        Returns:
            Refined image as RGB numpy uint8 array.
        """
        self._load_ip_adapter()

        use_lcm: bool = config.get("use_lcm", False)
        if use_lcm:
            self._apply_lcm_scheduler()

        if depth_map is None:
            depth_map = self._estimate_depth(image)
        if depth_map.size != image.size:
            depth_map = depth_map.resize(image.size)

        ip_scale = float(config.get("ip_adapter_scale", 0.5))
        self._pipe.set_ip_adapter_scale(ip_scale)

        seed = config.get("seed")
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self._device).manual_seed(int(seed))

        num_steps = 4 if use_lcm else int(config.get("num_inference_steps", 30))

        output = self._pipe(
            prompt=config.get("prompt", "photorealistic, high quality"),
            negative_prompt=config.get(
                "negative_prompt", "blurry, distorted, artifacts, watermark"
            ),
            image=image,
            control_image=depth_map,
            ip_adapter_image=reference_image,
            strength=float(config.get("strength", 0.4)),
            controlnet_conditioning_scale=float(
                config.get("controlnet_conditioning_scale", 0.8)
            ),
            guidance_scale=float(config.get("guidance_scale", 7.5)),
            num_inference_steps=num_steps,
            generator=generator,
        )

        return np.array(output.images[0])

    @torch.no_grad()
    def _refine_lora(
        self,
        image,
        depth_map,
        config: dict,
    ) -> np.ndarray:
        """LoRA-conditioned img2img refinement.

        Swaps LoRA weights when the requested model differs from the currently
        loaded one.

        Args:
            image: PIL.Image RGB source image.
            depth_map: PIL.Image depth map, or None.
            config: DiffusionRefinementConfig fields as dict.

        Returns:
            Refined image as RGB numpy uint8 array.
        """
        lora_model_id: Optional[str] = config.get("lora_model_id")
        if not lora_model_id:
            raise ValueError("lora_model_id must be specified for LoRA refinement")

        # Ensure pipeline is ready
        if self._pipe is None:
            self._load_controlnet_pipeline()

        # Hot-swap LoRA if a different model is requested
        if self._loaded_lora_id != lora_model_id:
            if self._loaded_lora_id is not None:
                self._unload_lora()
            self._load_lora(lora_model_id)

        use_lcm: bool = config.get("use_lcm", False)
        if use_lcm:
            self._apply_lcm_scheduler()

        if depth_map is None:
            depth_map = self._estimate_depth(image)
        if depth_map.size != image.size:
            depth_map = depth_map.resize(image.size)

        lora_weight = float(config.get("lora_weight", 0.7))
        try:
            self._pipe.fuse_lora(lora_scale=lora_weight)
            fused = True
        except Exception:
            fused = False

        seed = config.get("seed")
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self._device).manual_seed(int(seed))

        num_steps = 4 if use_lcm else int(config.get("num_inference_steps", 30))

        output = self._pipe(
            prompt=config.get("prompt", "photorealistic, high quality"),
            negative_prompt=config.get(
                "negative_prompt", "blurry, distorted, artifacts, watermark"
            ),
            image=image,
            control_image=depth_map,
            strength=float(config.get("strength", 0.4)),
            controlnet_conditioning_scale=float(
                config.get("controlnet_conditioning_scale", 0.8)
            ),
            guidance_scale=float(config.get("guidance_scale", 7.5)),
            num_inference_steps=num_steps,
            generator=generator,
        )

        if fused:
            try:
                self._pipe.unfuse_lora()
            except Exception:
                pass

        return np.array(output.images[0])

    # =========================================================================
    # Batch Refinement
    # =========================================================================

    def refine_batch(
        self,
        images_dir: str,
        output_dir: str,
        config: dict,
        reference_dir: Optional[str] = None,
        annotations_dir: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
    ) -> dict:
        """Refine all images in a directory.

        Searches for depth maps using the ``_find_depth_map`` helper and
        annotation files by matching stem + extension in ``annotations_dir``.
        Calls ``progress_callback(processed, total)`` after each image when
        the callback is supplied.

        Args:
            images_dir: Directory containing synthetic images to refine.
            output_dir: Directory for refined output images.
            config: DiffusionRefinementConfig fields as dict.
            reference_dir: Reference images directory for IP-Adapter.
            annotations_dir: Directory containing annotation files.
            progress_callback: Optional callable(processed: int, total: int).

        Returns:
            Dict with ``total``, ``processed``, ``failed``, and
            ``avg_preservation`` (averaged annotation metrics or None).
        """
        if not os.path.isdir(images_dir):
            logger.error("Images directory not found: {}", images_dir)
            return {"total": 0, "processed": 0, "failed": 0, "avg_preservation": None}

        image_files = sorted(
            p for p in Path(images_dir).iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not image_files:
            logger.warning("No images found in {}", images_dir)
            return {"total": 0, "processed": 0, "failed": 0, "avg_preservation": None}

        os.makedirs(output_dir, exist_ok=True)

        processed = 0
        failed = 0
        preservation_accumulator: Dict[str, List[float]] = {
            "edge_iou": [],
            "ssim": [],
            "mean_keypoint_displacement": [],
        }

        for idx, img_path in enumerate(image_files):
            output_path = os.path.join(output_dir, img_path.name)

            # Find depth map
            depth_map_path = self._find_depth_map(images_dir, img_path.stem)

            # Find annotation file
            ann_path: Optional[str] = None
            if annotations_dir:
                for ann_ext in [".json", ".txt", ".xml"]:
                    candidate = Path(annotations_dir) / f"{img_path.stem}{ann_ext}"
                    if candidate.exists():
                        ann_path = str(candidate)
                        break

            result = self.refine_single(
                image_path=str(img_path),
                output_path=output_path,
                config=config,
                depth_map_path=depth_map_path,
                reference_dir=reference_dir,
                annotations_path=ann_path,
            )

            if result["success"]:
                processed += 1
                pm = result.get("preservation_metrics")
                if pm:
                    for key in preservation_accumulator:
                        val = pm.get(key)
                        if val is not None:
                            preservation_accumulator[key].append(float(val))
            else:
                logger.error(
                    "refine_batch: failed on {} — {}", img_path.name, result["error"]
                )
                failed += 1

            if progress_callback:
                progress_callback(idx + 1, len(image_files))

        # Compute average preservation metrics
        avg_preservation: Optional[dict] = None
        if any(preservation_accumulator.values()):
            avg_preservation = {}
            for key, values in preservation_accumulator.items():
                avg_preservation[key] = float(np.mean(values)) if values else None

        # Release GPU after batch
        self._free_gpu_memory()

        logger.info(
            "refine_batch complete: total={} processed={} failed={}",
            len(image_files),
            processed,
            failed,
        )

        return {
            "total": len(image_files),
            "processed": processed,
            "failed": failed,
            "avg_preservation": avg_preservation,
        }

    # =========================================================================
    # LoRA Training
    # =========================================================================

    def train_lora(
        self,
        reference_dir: str,
        model_id: str,
        config: dict,
        progress_callback: Optional[Callable] = None,
    ) -> dict:
        """Train a LoRA adapter on real reference images using DreamBooth-style training.

        This method requires gradient computation and must NOT be decorated with
        ``@torch.no_grad()``.

        The training loop:
        1. Collects reference images from ``reference_dir``.
        2. Loads the SD 1.5 UNet and applies a PEFT LoRA configuration.
        3. Runs a simplified training loop with AdamW and cosine LR decay.
        4. Saves adapter weights plus ``metadata.json`` to
           ``self._lora_dir / model_id /``.

        Args:
            reference_dir: Directory of real reference images for fine-tuning.
            model_id: Unique name for the resulting LoRA model.
            config: Dict representation of LoRATrainingConfig fields.
            progress_callback: Optional callable(step: int, total: int).

        Returns:
            Dict with keys: ``success``, ``model_id``, ``model_dir``,
            ``training_steps``, ``training_time_seconds``, ``model_size_mb``,
            ``error``.
        """
        start_time = time.time()

        def _error(msg: str) -> dict:
            elapsed = time.time() - start_time
            logger.error("train_lora [{}] failed: {}", model_id, msg)
            return {
                "success": False,
                "model_id": model_id,
                "model_dir": "",
                "training_steps": 0,
                "training_time_seconds": elapsed,
                "model_size_mb": 0.0,
                "error": msg,
            }

        # --- Validate reference directory -------------------------------------
        if not os.path.isdir(reference_dir):
            return _error(f"Reference directory not found: {reference_dir}")

        ref_images = sorted(
            str(p) for p in Path(reference_dir).iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not ref_images:
            return _error(f"No images found in reference directory: {reference_dir}")

        # --- Check optional dependencies -------------------------------------
        try:
            from diffusers import UNet2DConditionModel, DDPMScheduler
            from transformers import CLIPTextModel, CLIPTokenizer
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            return _error(
                f"Required packages missing ({exc}). "
                "Install with: pip install diffusers transformers peft accelerate"
            )

        # --- Config -----------------------------------------------------------
        training_steps: int = int(config.get("training_steps", 500))
        learning_rate: float = float(config.get("learning_rate", 1e-4))
        lora_rank: int = int(config.get("lora_rank", 4))
        resolution: int = int(config.get("resolution", 512))
        batch_size: int = int(config.get("batch_size", 1))
        prompt_template: str = config.get("prompt_template", "a photo in the style of {domain}")
        reference_set_id: str = config.get("reference_set_id", "unknown")
        prompt: str = prompt_template.replace("{domain}", reference_set_id)

        model_dir = self._lora_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Starting LoRA training: model_id={} steps={} rank={} device={}",
            model_id,
            training_steps,
            lora_rank,
            self._device,
        )

        # Ensure inference pipeline is not occupying VRAM during training
        if self._pipe is not None:
            logger.info("Releasing inference pipeline VRAM before LoRA training")
            self._free_gpu_memory()

        try:
            dtype = torch.float16 if self._device.type == "cuda" else torch.float32

            # Load tokenizer and text encoder for prompt conditioning
            tokenizer = CLIPTokenizer.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                subfolder="tokenizer",
                cache_dir=str(self._models_dir),
            )
            text_encoder = CLIPTextModel.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                subfolder="text_encoder",
                torch_dtype=dtype,
                cache_dir=str(self._models_dir),
            ).to(self._device)
            text_encoder.requires_grad_(False)

            # Load UNet and apply LoRA
            unet = UNet2DConditionModel.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                subfolder="unet",
                torch_dtype=dtype,
                cache_dir=str(self._models_dir),
            )

            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank * 2,
                target_modules=["to_q", "to_v", "to_k", "to_out.0"],
                lora_dropout=0.05,
                bias="none",
            )
            unet = get_peft_model(unet, lora_config)
            unet.to(self._device)
            unet.train()

            # Noise scheduler
            noise_scheduler = DDPMScheduler.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                subfolder="scheduler",
                cache_dir=str(self._models_dir),
            )

            # Optimizer (only LoRA parameters)
            lora_params = [p for p in unet.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(lora_params, lr=learning_rate)

            # Cosine LR scheduler
            from torch.optim.lr_scheduler import CosineAnnealingLR
            lr_scheduler = CosineAnnealingLR(
                optimizer, T_max=training_steps, eta_min=learning_rate * 0.1
            )

            # Encode prompt once
            with torch.no_grad():
                text_inputs = tokenizer(
                    [prompt] * batch_size,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                encoder_hidden_states = text_encoder(
                    text_inputs.input_ids.to(self._device)
                )[0]

            # Training loop
            from PIL import Image as PILImage
            step = 0
            epoch_images = self._load_training_images(ref_images, resolution)

            while step < training_steps:
                # Shuffle images each pass
                random.shuffle(epoch_images)

                for batch_start in range(0, len(epoch_images), batch_size):
                    if step >= training_steps:
                        break

                    batch_tensors = epoch_images[batch_start: batch_start + batch_size]
                    # Pad batch to batch_size if needed
                    while len(batch_tensors) < batch_size:
                        batch_tensors = batch_tensors + [batch_tensors[-1]]

                    pixel_values = torch.stack(batch_tensors).to(
                        dtype=dtype, device=self._device
                    )

                    # Sample noise and timesteps
                    noise = torch.randn_like(pixel_values)
                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (pixel_values.shape[0],),
                        device=self._device,
                    ).long()

                    # Add noise
                    noisy_latents = noise_scheduler.add_noise(pixel_values, noise, timesteps)

                    # Forward pass
                    noise_pred = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states[: pixel_values.shape[0]],
                    ).sample

                    loss = torch.nn.functional.mse_loss(noise_pred, noise)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
                    optimizer.step()
                    lr_scheduler.step()

                    step += 1

                    if step % 50 == 0 or step == training_steps:
                        current_lr = optimizer.param_groups[0]["lr"]
                        logger.info(
                            "LoRA training [{}/{}] loss={:.4f} lr={:.2e}",
                            step,
                            training_steps,
                            loss.item(),
                            current_lr,
                        )

                    if progress_callback:
                        progress_callback(step, training_steps)

            # Save LoRA adapter weights in diffusers-compatible format
            # (matches pipe.load_lora_weights() used at inference time)
            self._pipe.save_lora_weights(
                save_directory=str(model_dir),
                unet_lora_layers=unet,
            )

            # Measure saved size
            model_size_mb = sum(
                f.stat().st_size for f in model_dir.rglob("*") if f.is_file()
            ) / (1024 ** 2)

            training_time = time.time() - start_time

            # Save metadata
            metadata = {
                "model_id": model_id,
                "reference_set_id": reference_set_id,
                "training_steps": training_steps,
                "learning_rate": learning_rate,
                "lora_rank": lora_rank,
                "resolution": resolution,
                "batch_size": batch_size,
                "prompt": prompt,
                "model_size_mb": round(model_size_mb, 2),
                "training_time_seconds": round(training_time, 2),
                "device": str(self._device),
                "created_at": datetime.now().isoformat(),
            }
            with open(model_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(
                "LoRA training complete: model_id={} steps={} time={:.1f}s size={:.1f}MB",
                model_id,
                training_steps,
                training_time,
                model_size_mb,
            )

            return {
                "success": True,
                "model_id": model_id,
                "model_dir": str(model_dir),
                "training_steps": training_steps,
                "training_time_seconds": round(training_time, 2),
                "model_size_mb": round(model_size_mb, 2),
                "error": None,
            }

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                logger.error(
                    "CUDA out of memory during LoRA training. "
                    "Try reducing batch_size or resolution. Error: {}",
                    exc,
                )
            return _error(str(exc))
        except Exception as exc:
            return _error(str(exc))
        finally:
            # Clean up training objects from VRAM
            try:
                del unet, text_encoder
            except Exception:
                pass
            self._free_gpu_memory()

    @staticmethod
    def _load_training_images(
        image_paths: List[str], resolution: int
    ) -> List[torch.Tensor]:
        """Load and preprocess training images as normalized tensors.

        Args:
            image_paths: List of file paths.
            resolution: Target square resolution.

        Returns:
            List of (3, resolution, resolution) float32 tensors in [-1, 1].
        """
        from PIL import Image as PILImage

        tensors: List[torch.Tensor] = []
        for path in image_paths:
            try:
                img = PILImage.open(path).convert("RGB")
                img = img.resize((resolution, resolution), PILImage.LANCZOS)
                arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
                tensor = torch.from_numpy(arr).permute(2, 0, 1)
                tensors.append(tensor)
            except Exception as exc:
                logger.warning("Skipping training image {}: {}", path, exc)
        return tensors

    # =========================================================================
    # LoRA Model Management
    # =========================================================================

    def list_lora_models(self) -> List[dict]:
        """List all trained LoRA models found in ``self._lora_dir``.

        Only directories that contain a ``metadata.json`` file are returned.

        Returns:
            List of dicts with LoRA model metadata.
        """
        models: List[dict] = []

        if not self._lora_dir.is_dir():
            return models

        for entry in sorted(self._lora_dir.iterdir()):
            if not entry.is_dir():
                continue
            meta_path = entry / "metadata.json"
            if not meta_path.is_file():
                continue
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                models.append(meta)
            except Exception as exc:
                logger.warning("Failed to read LoRA metadata for {}: {}", entry.name, exc)

        logger.debug("Found {} LoRA models", len(models))
        return models

    def delete_lora_model(self, model_id: str) -> bool:
        """Delete a LoRA model and all its files.

        Unloads the model from the pipeline first when it is currently active.

        Args:
            model_id: Identifier of the LoRA model to remove.

        Returns:
            True if deleted; False if the model directory was not found.
        """
        model_path = self._lora_dir / model_id

        if not model_path.is_dir():
            logger.warning("LoRA model not found for deletion: {}", model_id)
            return False

        if self._loaded_lora_id == model_id:
            self._unload_lora()

        shutil.rmtree(str(model_path))
        logger.info("Deleted LoRA model: {}", model_id)
        return True

    # =========================================================================
    # Annotation Preservation Validation
    # =========================================================================

    @staticmethod
    def validate_annotations(
        original: np.ndarray,
        refined: np.ndarray,
        threshold: float = 0.7,
    ) -> dict:
        """Compute annotation preservation metrics between two RGB images.

        Args:
            original: Original synthetic RGB image as uint8 numpy array.
            refined: Refined RGB image as uint8 numpy array.
            threshold: Minimum Edge IoU value for ``annotations_valid``.

        Returns:
            Dict with keys ``edge_iou``, ``ssim``,
            ``mean_keypoint_displacement``, ``annotations_valid``.
        """
        # Ensure uint8
        if original.dtype != np.uint8:
            original = np.clip(original, 0, 255).astype(np.uint8)
        if refined.dtype != np.uint8:
            refined = np.clip(refined, 0, 255).astype(np.uint8)

        # Resize refined to match original if shapes differ
        if original.shape[:2] != refined.shape[:2]:
            refined = cv2.resize(
                refined,
                (original.shape[1], original.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        edge_iou = DiffusionRefinementEngine._compute_edge_iou(original, refined)
        ssim = DiffusionRefinementEngine._compute_ssim(original, refined)
        mean_kp_disp = DiffusionRefinementEngine._compute_keypoint_displacement(
            original, refined
        )

        return {
            "edge_iou": edge_iou,
            "ssim": ssim,
            "mean_keypoint_displacement": mean_kp_disp,
            "annotations_valid": edge_iou >= threshold,
        }

    @staticmethod
    def _compute_edge_iou(img1: np.ndarray, img2: np.ndarray) -> float:
        """Canny edge IoU between two RGB images.

        Both images are converted to grayscale before edge detection.

        Args:
            img1: First RGB uint8 image.
            img2: Second RGB uint8 image.

        Returns:
            Edge IoU in [0, 1] (1 = identical edges).
        """
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) if img1.ndim == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) if img2.ndim == 3 else img2

        edges1 = cv2.Canny(gray1, 50, 150)
        edges2 = cv2.Canny(gray2, 50, 150)

        intersection = np.logical_and(edges1 > 0, edges2 > 0).sum()
        union = np.logical_or(edges1 > 0, edges2 > 0).sum()
        return float(intersection / max(union, 1))

    @staticmethod
    def _compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
        """Structural Similarity Index between two RGB images.

        Args:
            img1: First RGB uint8 image.
            img2: Second RGB uint8 image.

        Returns:
            SSIM in [-1, 1] (1 = identical).
        """
        try:
            from skimage.metrics import structural_similarity

            return float(
                structural_similarity(
                    img1,
                    img2,
                    channel_axis=2 if img1.ndim == 3 else None,
                    data_range=255,
                )
            )
        except ImportError:
            logger.warning(
                "scikit-image not installed; SSIM will return 0.0. "
                "Install with: pip install scikit-image"
            )
            return 0.0
        except Exception as exc:
            logger.warning("SSIM computation failed: {}", exc)
            return 0.0

    @staticmethod
    def _compute_keypoint_displacement(
        img1: np.ndarray, img2: np.ndarray
    ) -> float:
        """Mean ORB keypoint displacement between two images.

        Detects ORB keypoints, matches them with brute-force Hamming distance,
        and computes the mean Euclidean displacement of matched pairs.

        Args:
            img1: First RGB uint8 image.
            img2: Second RGB uint8 image.

        Returns:
            Mean displacement in pixels (0.0 when no matches are found).
        """
        orb = cv2.ORB_create(nfeatures=500)

        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) if img1.ndim == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) if img2.ndim == 3 else img2

        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)

        if des1 is None or des2 is None:
            return 0.0

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        if not matches:
            return 0.0

        displacements = [
            np.sqrt(
                (kp1[m.queryIdx].pt[0] - kp2[m.trainIdx].pt[0]) ** 2
                + (kp1[m.queryIdx].pt[1] - kp2[m.trainIdx].pt[1]) ** 2
            )
            for m in matches
        ]
        return float(np.mean(displacements))

    # =========================================================================
    # Helpers
    # =========================================================================

    @staticmethod
    def _pick_random_reference(reference_dir: str) -> str:
        """Choose a random image file from ``reference_dir``.

        Args:
            reference_dir: Directory containing reference images.

        Returns:
            Absolute path to a randomly selected image.

        Raises:
            FileNotFoundError: When the directory does not exist.
            ValueError: When no valid images are found.
        """
        dir_path = Path(reference_dir)
        if not dir_path.is_dir():
            raise FileNotFoundError(
                f"Reference directory not found: {reference_dir}"
            )

        candidates = sorted(
            p for p in dir_path.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not candidates:
            raise ValueError(
                f"No valid reference images found in: {reference_dir}"
            )

        return str(random.choice(candidates))

    @staticmethod
    def _find_depth_map(images_dir: str, stem: str) -> Optional[str]:
        """Search for a depth map matching ``stem`` near the images directory.

        Search order:
        1. ``{images_dir}/{stem}_depth.{ext}`` for ext in [.png, .jpg, .tiff, .exr].
        2. ``{images_dir}/depth/{stem}.{ext}`` subdirectory.

        Args:
            images_dir: Directory containing the source images.
            stem: File stem (name without extension) of the source image.

        Returns:
            Absolute path to the depth map if found, else None.
        """
        base = Path(images_dir)

        for ext in [".png", ".jpg", ".tiff", ".exr"]:
            candidate = base / f"{stem}_depth{ext}"
            if candidate.is_file():
                return str(candidate)

        depth_subdir = base / "depth"
        if depth_subdir.is_dir():
            for ext in [".png", ".jpg", ".tiff", ".exr"]:
                candidate = depth_subdir / f"{stem}{ext}"
                if candidate.is_file():
                    return str(candidate)

        return None

    def _free_gpu_memory(self) -> None:
        """Release all VRAM held by the diffusion pipeline.

        Sets ``_pipe``, ``_controlnet``, ``_ip_adapter_loaded``, and
        ``_loaded_lora_id`` back to their initial state before calling
        ``torch.cuda.empty_cache()``.
        """
        if self._pipe is not None:
            try:
                del self._pipe
            except Exception:
                pass
            self._pipe = None

        self._controlnet = None
        self._ip_adapter_loaded = False
        self._loaded_lora_id = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("GPU memory released (empty_cache called)")

    # =========================================================================
    # Internal Utilities
    # =========================================================================

    def _apply_lcm_scheduler(self) -> None:
        """Replace the pipeline scheduler with LCMScheduler for 4-step generation.

        No-op when the pipeline is not loaded or when LCMScheduler is already
        active.
        """
        if self._pipe is None:
            return
        try:
            from diffusers import LCMScheduler

            if not isinstance(self._pipe.scheduler, LCMScheduler):
                self._pipe.scheduler = LCMScheduler.from_config(
                    self._pipe.scheduler.config
                )
                logger.info("LCM scheduler applied for fast 4-step generation")
        except ImportError:
            logger.warning(
                "LCMScheduler not available in installed diffusers version; "
                "using default scheduler"
            )

    @staticmethod
    def _load_depth_pil(depth_path: str, target_size: tuple):
        """Load a depth map and return it as a grayscale PIL.Image.

        Args:
            depth_path: Path to the depth map image.
            target_size: (width, height) tuple for resizing.

        Returns:
            Grayscale PIL.Image resized to ``target_size``.
        """
        from PIL import Image as PILImage

        depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        if depth is None:
            raise FileNotFoundError(f"Could not read depth map: {depth_path}")

        pil_depth = PILImage.fromarray(depth, mode="L")
        if pil_depth.size != target_size:
            pil_depth = pil_depth.resize(target_size, PILImage.LANCZOS)
        return pil_depth

    def _estimate_depth(self, image):
        """Estimate a depth map for ``image`` using MiDaS via controlnet_aux.

        Falls back to a uniform mid-grey PIL image when ``controlnet_aux`` is
        not installed, which is sufficient for ControlNet to run (with reduced
        structural conditioning).

        Args:
            image: PIL.Image RGB source image.

        Returns:
            Grayscale PIL.Image depth map of the same size as ``image``.
        """
        try:
            from controlnet_aux import MidasDetector

            midas = MidasDetector.from_pretrained("lllyasviel/Annotators")
            depth = midas(image)
            logger.debug("Depth map estimated with MiDaS")
            return depth
        except ImportError:
            logger.warning(
                "controlnet_aux not installed; using uniform depth map. "
                "Install with: pip install controlnet-aux"
            )
        except Exception as exc:
            logger.warning(
                "MiDaS depth estimation failed ({}); using uniform depth map", exc
            )

        from PIL import Image as PILImage

        return PILImage.new("L", image.size, color=128)
