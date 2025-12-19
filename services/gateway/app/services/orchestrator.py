"""
Pipeline Orchestrator

Coordinates the synthetic data generation pipeline by calling
the appropriate services in sequence.
"""
import os
import time
import logging
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional

from app.services.client import get_service_registry
from app.models.schemas import (
    GenerationConfig,
    ObjectInfo,
    AnnotationBox,
    EffectType
)

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrates the synthetic image generation pipeline."""

    def __init__(self):
        """Initialize orchestrator."""
        self.registry = get_service_registry()
        self.shared_dir = Path("/shared")

    async def generate_single_image(
        self,
        background_path: str,
        objects: List[ObjectInfo],
        config: GenerationConfig,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate a single synthetic image.

        Pipeline:
        1. Depth estimation on background (if depth_aware)
        2. Apply effects to result
        3. Generate annotations

        Args:
            background_path: Path to background image
            objects: List of objects to place
            config: Generation configuration
            output_path: Output path (auto-generated if None)

        Returns:
            Generation result dictionary
        """
        start_time = time.time()
        result = {
            "success": False,
            "output_path": "",
            "depth_map_path": None,
            "annotations": [],
            "objects_placed": 0,
            "effects_applied": [],
            "processing_time_ms": 0,
            "error": None
        }

        try:
            # Validate background exists
            bg_path = Path(background_path)
            if not bg_path.exists():
                raise FileNotFoundError(f"Background not found: {background_path}")

            # Generate output path if not provided
            if output_path is None:
                output_dir = self.shared_dir / "output" / "images"
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = str(output_dir / f"synthetic_{uuid.uuid4().hex[:8]}.jpg")

            result["output_path"] = output_path

            # Step 1: Depth estimation (if enabled)
            depth_map_path = None
            if config.depth_aware:
                try:
                    depth_result = await self.registry.depth.post("/estimate", {
                        "input_path": background_path,
                        "output_dir": str(self.shared_dir / "depth"),
                        "normalize": True,
                        "generate_preview": True,
                        "classify_zones": True
                    })

                    if depth_result.get("success"):
                        depth_map_path = depth_result.get("depth_map_path")
                        result["depth_map_path"] = depth_map_path
                        logger.info(f"Depth estimation completed: {depth_map_path}")
                    else:
                        logger.warning(f"Depth estimation failed: {depth_result.get('error')}")

                except Exception as e:
                    logger.warning(f"Depth service error: {e}. Continuing without depth.")

            # Step 2: Apply effects
            effects_to_apply = [e.value for e in config.effects]
            effects_applied = []

            try:
                effects_result = await self.registry.effects.post("/apply", {
                    "background_path": background_path,
                    "depth_map_path": depth_map_path,
                    "output_path": output_path,
                    "effects": effects_to_apply,
                    "config": {
                        "color_intensity": config.color_intensity,
                        "underwater_intensity": config.underwater_intensity,
                        "caustics_intensity": config.caustics_intensity
                    }
                })

                if effects_result.get("success"):
                    effects_applied = effects_result.get("effects_applied", [])
                    result["effects_applied"] = effects_applied
                    result["output_path"] = effects_result.get("output_path", output_path)
                    logger.info(f"Effects applied: {effects_applied}")
                else:
                    logger.warning(f"Effects failed: {effects_result.get('error')}")

            except Exception as e:
                logger.warning(f"Effects service error: {e}")

            # Step 3: Generate annotations for placed objects
            annotations = []
            for i, obj in enumerate(objects):
                # In a full implementation, this would track actual placement positions
                # For now, create placeholder annotations
                if obj.position:
                    x, y = obj.position
                else:
                    x, y = 100 + i * 50, 100 + i * 50

                annotations.append(AnnotationBox(
                    x=x,
                    y=y,
                    width=100,  # Placeholder
                    height=100,  # Placeholder
                    class_name=obj.class_name
                ))

            result["annotations"] = annotations
            result["objects_placed"] = len(objects)
            result["success"] = True

        except FileNotFoundError as e:
            result["error"] = str(e)
            logger.error(f"File not found: {e}")

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Pipeline error: {e}")

        result["processing_time_ms"] = (time.time() - start_time) * 1000
        return result

    async def generate_batch(
        self,
        backgrounds_dir: str,
        objects_dir: str,
        output_dir: str,
        num_images: int,
        config: GenerationConfig
    ) -> Dict[str, Any]:
        """Generate a batch of synthetic images.

        Args:
            backgrounds_dir: Directory containing background images
            objects_dir: Directory containing object images
            output_dir: Output directory
            num_images: Number of images to generate
            config: Generation configuration

        Returns:
            Batch generation result
        """
        start_time = time.time()
        job_id = uuid.uuid4().hex[:12]

        result = {
            "success": False,
            "job_id": job_id,
            "output_dir": output_dir,
            "total_requested": num_images,
            "generated": 0,
            "failed": 0,
            "processing_time_ms": 0,
            "error": None
        }

        try:
            # Validate directories
            bg_dir = Path(backgrounds_dir)
            obj_dir = Path(objects_dir)
            out_dir = Path(output_dir)

            if not bg_dir.exists():
                raise FileNotFoundError(f"Backgrounds_filtered directory not found: {backgrounds_dir}")

            out_dir.mkdir(parents=True, exist_ok=True)

            # Find background images
            bg_files = list(bg_dir.glob("*.jpg")) + list(bg_dir.glob("*.png"))
            if not bg_files:
                raise ValueError(f"No images found in backgrounds directory")

            # Find object images
            obj_files = []
            if obj_dir.exists():
                obj_files = list(obj_dir.glob("*.png")) + list(obj_dir.glob("*.jpg"))

            # Generate images
            import random

            for i in range(num_images):
                try:
                    # Select random background
                    bg_path = str(random.choice(bg_files))

                    # Select random objects (if available)
                    objects = []
                    if obj_files:
                        num_objs = random.randint(1, min(config.max_objects, len(obj_files)))
                        selected_objs = random.sample(obj_files, num_objs)
                        objects = [
                            ObjectInfo(
                                image_path=str(obj),
                                class_name=obj.stem.split('_')[0]  # Extract class from filename
                            )
                            for obj in selected_objs
                        ]

                    # Generate output path
                    output_path = str(out_dir / f"{job_id}_{i:04d}.jpg")

                    # Generate image
                    gen_result = await self.generate_single_image(
                        background_path=bg_path,
                        objects=objects,
                        config=config,
                        output_path=output_path
                    )

                    if gen_result.get("success"):
                        result["generated"] += 1
                    else:
                        result["failed"] += 1
                        logger.warning(f"Image {i} failed: {gen_result.get('error')}")

                except Exception as e:
                    result["failed"] += 1
                    logger.error(f"Image {i} error: {e}")

            result["success"] = result["failed"] == 0
            result["processing_time_ms"] = (time.time() - start_time) * 1000

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Batch generation error: {e}")

        return result


# Global orchestrator instance
_orchestrator: Optional[PipelineOrchestrator] = None


def get_orchestrator() -> PipelineOrchestrator:
    """Get singleton orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = PipelineOrchestrator()
    return _orchestrator
