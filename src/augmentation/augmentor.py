# src/augmentation/augmentor.py
import os
import json
import random
import math
import base64
import cv2
import numpy as np
import logging
import glob

from tqdm import tqdm
from collections import defaultdict

from src.utils.helpers import ensure_dir
from src.analysis.class_analysis import analyze_coco_dataset
from src.augmentation.transformations import rotate_image, scale_image, apply_perspective_transform, apply_motion_blur
from src.augmentation.realism import (apply_poisson_blending, transfer_color_correction,
                                      match_blur, generate_shadow, add_lighting_effect,
                                      apply_underwater_effect, add_upscaling_noise, generate_caustics_map, apply_caustics)
from src.augmentation.depth_engine import DepthEstimator
from src.augmentation.lighting_engine import (AdvancedLightingEstimator, generate_multi_source_shadows,
                                               apply_underwater_light_attenuation)
from src.validation.quality_metrics import QualityValidator, QualityScore, Anomaly
from src.validation.physics_validator import PhysicsValidator, PhysicsViolation

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

# Paleta de colores para dibujar las anotaciones
COLOR_PALETTE = [
    (0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255),
    (255, 255, 0), (0, 128, 255), (128, 0, 128), (60, 180, 75), (245, 130, 48),
    (0, 130, 200), (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60),
    (250, 190, 190), (0, 128, 128), (230, 190, 255), (170, 110, 40), (255, 250, 200),
    (128, 0, 0), (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128),
    (128, 128, 128)
]

def compute_iou(boxA, boxB):
    """Calcula la Intersección sobre la Unión (IoU) de dos bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = float(boxAArea + boxBArea - interArea)
    return interArea / unionArea if unionArea != 0 else 0

def check_mask_overlap(placement_mask, candidate_mask, candidate_pos, threshold=0.05):
    """Comprueba de forma robusta el solapamiento entre máscaras con validación completa.

    Args:
        placement_mask: Máscara de ocupación del fondo (numpy array 2D)
        candidate_mask: Máscara del objeto candidato (numpy array 2D)
        candidate_pos: Tupla (x, y) de posición del candidato
        threshold: Umbral de ratio de overlap (0.0-1.0)

    Returns:
        bool: True si el overlap excede el threshold

    Raises:
        ValueError: Si las máscaras tienen formato inválido
    """
    try:
        # ===== Validación de entradas =====
        if placement_mask is None or candidate_mask is None:
            logging.warning("Máscara None pasada a check_mask_overlap")
            return False

        if len(placement_mask.shape) != 2 or len(candidate_mask.shape) != 2:
            logging.error(f"Máscaras deben ser 2D: placement={placement_mask.shape}, "
                         f"candidate={candidate_mask.shape}")
            return False

        x, y = candidate_pos
        cand_h, cand_w = candidate_mask.shape
        bg_h, bg_w = placement_mask.shape

        # Validar dimensiones
        if cand_h == 0 or cand_w == 0:
            logging.warning("Máscara candidata tiene dimensiones cero")
            return False

        if bg_h == 0 or bg_w == 0:
            logging.warning("Máscara de placement tiene dimensiones cero")
            return False

        # ===== Calcular ROI con clipping seguro =====
        roi_x_start = max(0, x)
        roi_y_start = max(0, y)
        roi_x_end = min(bg_w, x + cand_w)
        roi_y_end = min(bg_h, y + cand_h)

        # Validar que hay superposición
        if roi_x_start >= roi_x_end or roi_y_start >= roi_y_end:
            return False  # Sin superposición

        # ===== Calcular coordenadas en máscara candidata =====
        cand_roi_x_start = max(0, roi_x_start - x)
        cand_roi_y_start = max(0, roi_y_start - y)
        cand_roi_x_end = cand_roi_x_start + (roi_x_end - roi_x_start)
        cand_roi_y_end = cand_roi_y_start + (roi_y_end - roi_y_start)

        # Validar índices dentro de límites
        cand_roi_x_end = min(cand_roi_x_end, cand_w)
        cand_roi_y_end = min(cand_roi_y_end, cand_h)

        # ===== Recortar ROIs =====
        roi_placement = placement_mask[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        roi_candidate = candidate_mask[cand_roi_y_start:cand_roi_y_end,
                                       cand_roi_x_start:cand_roi_x_end]

        # Validar que ambas ROIs tienen el mismo tamaño
        if roi_placement.shape != roi_candidate.shape:
            logging.warning(f"Shape mismatch en overlap check: "
                          f"placement={roi_placement.shape} vs candidate={roi_candidate.shape}")
            # Ajustar tamaños si es necesario
            min_h = min(roi_placement.shape[0], roi_candidate.shape[0])
            min_w = min(roi_placement.shape[1], roi_candidate.shape[1])
            roi_placement = roi_placement[:min_h, :min_w]
            roi_candidate = roi_candidate[:min_h, :min_w]

        # ===== Calcular overlap =====
        intersection = cv2.bitwise_and(roi_placement, roi_candidate)

        candidate_area = cv2.countNonZero(roi_candidate)
        if candidate_area == 0:
            return False  # Sin área visible

        overlap_area = cv2.countNonZero(intersection)
        overlap_ratio = overlap_area / candidate_area

        # Log para debugging si overlap es alto
        if overlap_ratio > 0.5:
            logging.debug(f"Alto overlap detectado: {overlap_ratio:.2%} "
                         f"({overlap_area}/{candidate_area} píxeles)")

        return overlap_ratio > threshold

    except Exception as e:
        logging.error(f"Error en check_mask_overlap: {e}", exc_info=True)
        # En caso de error, asumir overlap para evitar colocar objeto
        return True

def refine_object_mask(alpha_channel, mask_bin):
    """
    Refina la máscara binaria para suavizar los bordes dentados (antialiasing),
    preservando mejor la integración con el fondo.
    
    Args:
        alpha_channel: Canal alpha original de la imagen (0-255).
        mask_bin: Máscara binaria actual (0 o 255).
        
    Returns:
        tuple: (alpha_suavizado, mask_bin_refinada)
    """
    # 1. Si el objeto original ya tenía transparencia variable, usarla (blend suave)
    # Verificamos si hay valores intermedios significativos en el alpha original
    unique_vals = np.unique(alpha_channel)
    has_transparency = len(unique_vals) > 5  # Más que solo 0 y 255 y algo de ruido
    
    if has_transparency:
        # Usar el alpha original pero limpiando el ruido de fondo muy bajo
        # Normalizamos para asegurar rango 0-1 float para operaciones matemáticas
        alpha_refined = alpha_channel.astype(np.float32) / 255.0
        # Hard clip para eliminar ruido de fondo casi invisible
        alpha_refined[alpha_refined < 0.05] = 0.0
        # Re-binarizar la máscara solo para lógica de colisiones (bbox)
        _, mask_refined_bin = cv2.threshold((alpha_refined * 255).astype(np.uint8), 10, 255, cv2.THRESH_BINARY)
        return alpha_refined, mask_refined_bin

    # 2. Si el objeto no tiene transparencia (es binario puro), fabricar antialiasing
    # Aplicar desenfoque gaussiano suave solo al borde de la máscara
    mask_float = mask_bin.astype(np.float32) / 255.0
    
    # Kernel size 3x3 o 5x5 dependiendo del tamaño de la imagen, aquí usamos 3x3 suave
    blurred_mask = cv2.GaussianBlur(mask_float, (3, 3), 0)
    
    # Esto crea un gradiente en los bordes en lugar de un escalón 0->1
    return blurred_mask, mask_bin

class SyntheticDataAugmentor:
    def __init__(self,
                 output_dir="/app/synthetic_dataset",
                 rot=True, scale=True, trans=True,
                 poisson_blending=False,
                 advanced_color_correction=True,
                 blur_consistency=True,
                 add_shadows=True,
                 perspective_transform=True,
                 lighting_effects=False,
                 motion_blur=False,
                 underwater_effect=True,
                 try_count=3,
                 overlap_threshold=0.1,
                 seed=42,
                 save_intermediate_steps=False,
                 realism_intensity=0.7,
                 max_upscale_ratio=4.0,
                 min_area_ratio=0.005,
                 max_area_ratio=0.4,
                 depth_aware=True,
                 depth_model_size='small',
                 depth_cache_dir='checkpoints',
                 # Advanced lighting parameters
                 advanced_lighting_enabled=False,
                 advanced_lighting_config=None,
                 # Validation parameters
                 validation_enabled=False,
                 validation_config=None):
        self.rot = rot
        self.scale = scale
        self.trans = trans
        self.poisson_blending = poisson_blending
        self.advanced_color_correction = advanced_color_correction
        self.blur_consistency = blur_consistency
        self.add_shadows = add_shadows
        self.perspective_transform = perspective_transform
        self.lighting_effects = lighting_effects
        self.motion_blur = motion_blur
        self.underwater_effect = underwater_effect
        self.try_count = try_count
        self.overlap_threshold = overlap_threshold
        self.output_dir = output_dir
        random.seed(seed)
        np.random.seed(seed)
        self.save_intermediate_steps = save_intermediate_steps
        self.realism_intensity = realism_intensity
        self.max_upscale_ratio = max_upscale_ratio

        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio

        # Depth-aware augmentation
        self.depth_aware = depth_aware
        self.depth_estimator = None
        if self.depth_aware:
            try:
                self.depth_estimator = DepthEstimator(
                    model_size=depth_model_size,
                    device='cuda',
                    cache_dir=depth_cache_dir
                )
                logging.info(f"Depth-aware augmentation enabled with {depth_model_size} model")
            except Exception as e:
                logging.warning(f"Failed to initialize depth estimator: {e}. Falling back to random scaling.")
                self.depth_aware = False

        # Quality validation system
        self.validation_enabled = validation_enabled
        self.validation_config = validation_config or {}
        self.quality_validator = None
        self.physics_validator = None
        self.rejected_images = []

        if self.validation_enabled:
            try:
                # Initialize quality validator
                metrics_config = self.validation_config.get('metrics', {})
                thresholds_config = self.validation_config.get('thresholds', {})

                self.quality_validator = QualityValidator(
                    reference_dataset_path=self.validation_config.get('reference_dataset_path'),
                    device='cuda',
                    use_lpips=metrics_config.get('lpips_enabled', True),
                    use_fid=metrics_config.get('fid_enabled', False),
                    use_anomaly_detection=metrics_config.get('anomaly_detection', True)
                )

                # Initialize physics validator
                physics_config = self.validation_config.get('physics', {})
                self.physics_validator = PhysicsValidator(
                    density_threshold_float=physics_config.get('density_threshold_float', 0.95),
                    density_threshold_sink=physics_config.get('density_threshold_sink', 1.15),
                    surface_zone=physics_config.get('surface_zone', 0.25),
                    bottom_zone=physics_config.get('bottom_zone', 0.75)
                )

                logging.info("Quality validation system enabled")

                # Create rejected images directory if needed
                if self.validation_config.get('save_rejected', True):
                    self.rejected_dir = os.path.join(
                        self.output_dir,
                        self.validation_config.get('rejected_dir', 'rejected_images')
                    )
                    ensure_dir(self.rejected_dir)

            except Exception as e:
                logging.warning(f"Failed to initialize validation system: {e}. Continuing without validation.")
                self.validation_enabled = False

        # Advanced lighting estimation system
        self.advanced_lighting_enabled = advanced_lighting_enabled
        self.advanced_lighting_config = advanced_lighting_config or {}
        self.lighting_estimator = None
        self.lighting_cache = {}  # Cache HDR environment maps per background

        if self.advanced_lighting_enabled:
            try:
                max_lights = self.advanced_lighting_config.get('max_light_sources', 3)
                intensity_threshold = self.advanced_lighting_config.get('intensity_threshold', 0.6)
                use_hdr = self.advanced_lighting_config.get('use_hdr_estimation', False)

                self.lighting_estimator = AdvancedLightingEstimator(
                    max_light_sources=max_lights,
                    intensity_threshold=intensity_threshold,
                    use_hdr_estimation=use_hdr
                )

                logging.info(f"Advanced lighting estimation enabled (max_lights={max_lights})")
            except Exception as e:
                logging.warning(f"Failed to initialize lighting estimator: {e}. Using basic shadows.")
                self.advanced_lighting_enabled = False

        self.images_dir = os.path.join(self.output_dir, "images")
        self.labels_dir = os.path.join(self.output_dir, "labels")
        self.annotated_dir = os.path.join(self.output_dir, "annotated")

        ensure_dir(self.images_dir)
        ensure_dir(self.labels_dir)
        ensure_dir(self.annotated_dir)

        if self.save_intermediate_steps:
            self.intermediate_dir = os.path.join(self.output_dir, "intermediate_steps")
            ensure_dir(self.intermediate_dir)
            self.step_dirs = {
                "01_background": os.path.join(self.intermediate_dir, "01_background"),
                "02_object_raw": os.path.join(self.intermediate_dir, "02_object_raw"),
                "03_pasted_raw": os.path.join(self.intermediate_dir, "03_pasted_raw"),
                "04_with_shadow": os.path.join(self.intermediate_dir, "04_with_shadow"),
                "05_realism_applied": os.path.join(self.intermediate_dir, "05_realism_applied"),
                "06_final_blended": os.path.join(self.intermediate_dir, "06_final_blended"),
            }
            for d in self.step_dirs.values():
                ensure_dir(d)
            logging.info(f"Guardado de pasos intermedios activado en: {self.intermediate_dir}")

    def apply_transformations(self, patch, bg_shape, original_dims):
        """Aplica una secuencia de transformaciones geométricas al objeto."""
        transformed_patch = patch.copy()
        h_bg, w_bg = bg_shape[:2]

        if self.perspective_transform:
            transformed_patch = apply_perspective_transform(transformed_patch)

        if self.rot:
            angle = random.uniform(-45, 45)
            transformed_patch = rotate_image(transformed_patch, angle)

        p_h, p_w = transformed_patch.shape[:2]
        if p_h == 0 or p_w == 0: return None, (0,0)

        if self.scale:
            max_allowed_scale = self.max_upscale_ratio
            max_scale_w = (w_bg * 0.7) / p_w
            max_scale_h = (h_bg * 0.7) / p_h
            max_scale = min(max_scale_w, max_scale_h, max_allowed_scale)
            min_scale = 0.1
            
            if max_scale > min_scale:
                 scale_factor = random.uniform(min_scale, max_scale)
                 transformed_patch = scale_image(transformed_patch, scale_factor)
                 if scale_factor > self.max_upscale_ratio * 0.75:
                     transformed_patch = add_upscaling_noise(transformed_patch)
            else:
                 logging.warning(f"El patch ya es muy grande para el fondo. No se aplica escalado.")

        p_h, p_w = transformed_patch.shape[:2]
        if p_h == 0 or p_w == 0: return None, (0,0)

        if self.trans:
            max_x = max(w_bg - p_w, 1)
            max_y = max(h_bg - p_h, 1)
            x, y = random.randrange(0, max_x), random.randrange(0, max_y)
        else:
            x, y = 0, 0
        pos = (x, y)

        return transformed_patch, pos

    def calculate_depth_aware_scale(self, obj_dims, bg_image, depth_map=None, target_position=None):
        """
        Calcula escala apropiada basada en profundidad del fondo.

        Solves user's problem: "en base al tamaño original de objeto y del fondo
        lo posicione sobre el fondo de tal manera que el objeto no se sobrepixele
        o sea muy pequeño"

        Args:
            obj_dims: (height, width) del objeto original
            bg_image: Imagen de fondo (numpy array)
            depth_map: Mapa de profundidad precalculado (opcional)
            target_position: Posición objetivo (x, y) o None para selección automática

        Returns:
            tuple: (scale_factor, position, depth_value, depth_map)
        """
        if not self.depth_aware or self.depth_estimator is None:
            # Fallback to random scaling
            h_bg, w_bg = bg_image.shape[:2]
            obj_h, obj_w = obj_dims
            max_scale_w = (w_bg * 0.7) / obj_w
            max_scale_h = (h_bg * 0.7) / obj_h
            max_scale = min(max_scale_w, max_scale_h, self.max_upscale_ratio)
            min_scale = 0.1
            scale_factor = random.uniform(min_scale, max_scale) if max_scale > min_scale else 1.0
            x = random.randint(0, max(w_bg - int(obj_w * scale_factor), 1))
            y = random.randint(0, max(h_bg - int(obj_h * scale_factor), 1))
            return scale_factor, (x, y), 0.5, None

        # Estimate depth map if not provided
        if depth_map is None:
            depth_map = self.depth_estimator.estimate_depth(bg_image, normalize=True)

        h_bg, w_bg = bg_image.shape[:2]
        obj_h, obj_w = obj_dims

        # Classify depth zones
        zones_mask, depth_ranges = self.depth_estimator.classify_depth_zones(depth_map, num_zones=3)

        # Define scale ranges for each depth zone
        # Zone 0 (near): Large objects (0.8 - 3.0x)
        # Zone 1 (mid): Medium objects (0.4 - 1.5x)
        # Zone 2 (far): Small objects (0.05 - 0.4x)
        scale_ranges = {
            0: (0.8, min(3.0, self.max_upscale_ratio)),  # Near
            1: (0.4, 1.5),  # Mid
            2: (0.05, 0.4)   # Far
        }

        if target_position is not None:
            # Use specified position
            x, y = target_position
            # Ensure position is valid
            x = max(0, min(x, w_bg - obj_w))
            y = max(0, min(y, h_bg - obj_h))

            # Get depth at position
            depth_value = depth_map[y, x]

            # Determine zone from depth value
            zone = np.argmin([abs(depth_value - r[0]) for r in depth_ranges])
        else:
            # Weighted random zone selection (prefer mid zones for better realism)
            zone_weights = [0.25, 0.5, 0.25]  # Near, Mid, Far
            zone = random.choices([0, 1, 2], weights=zone_weights, k=1)[0]

            # Find valid positions in selected zone
            zone_coords = np.argwhere(zones_mask == zone)

            if len(zone_coords) == 0:
                # Fallback to any zone if selected zone is empty
                logging.warning(f"Selected depth zone {zone} is empty, using fallback")
                zone = random.randint(0, 2)
                zone_coords = np.argwhere(zones_mask == zone)

                if len(zone_coords) == 0:
                    # Ultimate fallback: random position
                    x = random.randint(0, max(w_bg - obj_w, 1))
                    y = random.randint(0, max(h_bg - obj_h, 1))
                    depth_value = depth_map[y, x]
                else:
                    idx = random.randint(0, len(zone_coords) - 1)
                    y, x = zone_coords[idx]
                    depth_value = depth_map[y, x]
            else:
                # Select random position within zone
                idx = random.randint(0, len(zone_coords) - 1)
                y, x = zone_coords[idx]
                depth_value = depth_map[y, x]

        # Calculate scale based on depth and zone
        min_scale, max_scale = scale_ranges[zone]

        # Interpolate scale within zone range based on exact depth value
        zone_depth_min, zone_depth_max = depth_ranges[zone]
        if zone_depth_max > zone_depth_min:
            depth_ratio = (depth_value - zone_depth_min) / (zone_depth_max - zone_depth_min)
            depth_ratio = np.clip(depth_ratio, 0, 1)
        else:
            depth_ratio = 0.5

        # For near zones (low depth values): higher ratio -> larger scale
        # For far zones (high depth values): higher ratio -> smaller scale
        if zone == 0:  # Near
            scale_factor = min_scale + depth_ratio * (max_scale - min_scale)
        elif zone == 2:  # Far
            scale_factor = max_scale - depth_ratio * (max_scale - min_scale)
        else:  # Mid
            scale_factor = min_scale + depth_ratio * (max_scale - min_scale)

        # Ensure object fits in background
        max_scale_w = (w_bg * 0.95) / obj_w
        max_scale_h = (h_bg * 0.95) / obj_h
        scale_factor = min(scale_factor, max_scale_w, max_scale_h)

        # Adjust position to ensure scaled object fits
        scaled_w = int(obj_w * scale_factor)
        scaled_h = int(obj_h * scale_factor)
        x = max(0, min(x, w_bg - scaled_w))
        y = max(0, min(y, h_bg - scaled_h))

        logging.debug(f"Depth-aware scaling: zone={zone}, depth={depth_value:.3f}, "
                     f"scale={scale_factor:.2f}, pos=({x},{y})")

        return scale_factor, (x, y), depth_value, depth_map

    def paste_object(self, bg_image, patch, original_dims, existing_bboxes=None, placement_mask=None, depth_map=None, lighting_map=None):
        """Intenta pegar un objeto en el fondo aplicando realismo y controlando solapamiento."""
        if existing_bboxes is None: existing_bboxes = []
        if placement_mask is None: placement_mask = np.zeros(bg_image.shape[:2], dtype=np.uint8)
        bg_h, bg_w = bg_image.shape[:2]
        bg_area = bg_h * bg_w

        for i in range(self.try_count):
            # Apply depth-aware scaling if enabled
            if self.depth_aware and self.depth_estimator is not None:
                # Use depth-aware scaling instead of random transformations
                scale_factor, pos, depth_value, depth_map = self.calculate_depth_aware_scale(
                    original_dims, bg_image, depth_map=depth_map
                )

                # Apply transformations without scaling (scaling handled by depth-aware)
                transformed_patch = patch.copy()

                if self.perspective_transform:
                    transformed_patch = apply_perspective_transform(transformed_patch)

                if self.rot:
                    angle = random.uniform(-45, 45)
                    transformed_patch = rotate_image(transformed_patch, angle)

                # Apply depth-aware scale
                transformed_patch = scale_image(transformed_patch, scale_factor)

                # Add upscaling noise if needed
                if scale_factor > self.max_upscale_ratio * 0.75:
                    transformed_patch = add_upscaling_noise(transformed_patch)

                # Position is already determined by depth-aware calculation
                # pos is already set
            else:
                # Fallback to original random transformations
                transformed_patch, pos = self.apply_transformations(patch, bg_image.shape, original_dims)

            if transformed_patch is None or transformed_patch.shape[0] == 0 or transformed_patch.shape[1] == 0:
                continue

            if transformed_patch.shape[2] < 4:
                # Si no tiene alpha, creamos uno sólido
                alpha_channel = np.ones(transformed_patch.shape[:2], dtype=np.uint8) * 255
                obj_bgr = transformed_patch
                # Alpha para blending es todo 1.0
                alpha_factor = np.ones(transformed_patch.shape[:2], dtype=np.float32)
                _, mask_bin = cv2.threshold(alpha_channel, 127, 255, cv2.THRESH_BINARY)
            else:
                alpha_channel = transformed_patch[:, :, 3]
                obj_bgr = transformed_patch[:, :, :3]
                
                # Obtenemos un alpha float (0.0-1.0) suave y la máscara binaria para lógica
                # Usamos un umbral bajo inicial para no perder detalles finos antes de refinar
                _, temp_mask = cv2.threshold(alpha_channel, 10, 255, cv2.THRESH_BINARY)
                alpha_factor, mask_bin = refine_object_mask(alpha_channel, temp_mask)
                # CORRECCIÓN: Eliminada la línea que sobrescribía mask_bin con umbral 127

            object_area = cv2.countNonZero(mask_bin)
            if object_area == 0:
                continue # Objeto no visible
            
            object_ratio = object_area / bg_area
            if not (self.min_area_ratio <= object_ratio <= self.max_area_ratio):
                continue # Objeto demasiado grande o pequeño, intentar otra transformación

            if check_mask_overlap(placement_mask, mask_bin, (pos[0], pos[1]), threshold=self.overlap_threshold):
                continue

            x_obj_in_patch, y_obj_in_patch, w_obj_in_patch, h_obj_in_patch = cv2.boundingRect(mask_bin)
            candidate_box = (pos[0] + x_obj_in_patch, pos[1] + y_obj_in_patch,
                             pos[0] + x_obj_in_patch + w_obj_in_patch, pos[1] + y_obj_in_patch + h_obj_in_patch)

            if any(compute_iou(candidate_box, bb) > self.overlap_threshold for bb in existing_bboxes):
                continue

            bg_h, bg_w = bg_image.shape[:2]
            p_h, p_w = transformed_patch.shape[:2]
            x_paste, y_paste = pos

            x_start_bg, y_start_bg = max(x_paste, 0), max(y_paste, 0)
            x_end_bg, y_end_bg = min(x_paste + p_w, bg_w), min(y_paste + p_h, bg_h)
            x_start_patch, y_start_patch = max(0, -x_paste), max(0, -y_paste)
            eff_w, eff_h = x_end_bg - x_start_bg, y_end_bg - y_start_bg

            if eff_w <= 0 or eff_h <= 0: continue

            obj_bgr_eff = obj_bgr[y_start_patch:y_start_patch+eff_h, x_start_patch:x_start_patch+eff_w]
            mask_bin_eff = mask_bin[y_start_patch:y_start_patch+eff_h, x_start_patch:x_start_patch+eff_w]
            
            # Recortar también el alpha factor suavizado
            alpha_factor_eff = alpha_factor[y_start_patch:y_start_patch+eff_h, x_start_patch:x_start_patch+eff_w]
            # Necesitamos expandir dimensiones para multiplicar por los 3 canales de color (H, W, 1)
            alpha_factor_eff_3ch = alpha_factor_eff[:, :, np.newaxis]

            roi = bg_image[y_start_bg:y_end_bg, x_start_bg:x_end_bg]

            # PIPELINE ÓPTIMO DE EFECTOS (orden importa para realismo)
            obj_for_blending = obj_bgr_eff.copy()

            # 1. Color matching primero (base para otros efectos)
            if self.advanced_color_correction:
                obj_for_blending = transfer_color_correction(
                    obj_for_blending, roi, mask_bin_eff, intensity=self.realism_intensity
                )

            # 2. Blur consistency (antes de efectos de luz)
            if self.blur_consistency:
                obj_for_blending = match_blur(obj_for_blending, roi, mask_bin_eff)

            # 3. Lighting effects (modifican iluminación)
            if self.lighting_effects:
                light_type = 'spotlight' if np.random.random() > 0.5 else 'gradient'
                strength = np.random.uniform(1.2, 1.8)  # Variación aleatoria
                obj_for_blending = add_lighting_effect(obj_for_blending, light_type=light_type, strength=strength)

            # 4. Underwater effect (añade tinte global)
            if self.underwater_effect:
                water_tint = self._extract_water_tint(roi)
                obj_for_blending = apply_underwater_effect(
                    obj_for_blending,
                    color_cast=water_tint,
                    intensity=np.random.uniform(0.15, 0.35)  # Variación realista
                )

            # 5. Motion blur al FINAL (solo 20% de objetos para realismo)
            if self.motion_blur and np.random.random() < 0.2:
                kernel_size = random.randint(5, 15)
                angle = random.uniform(0, 360)
                obj_for_blending = apply_motion_blur(obj_for_blending, kernel_size, angle)

            # 6. Normalización final (prevenir valores extremos)
            obj_for_blending = np.clip(obj_for_blending, 0, 255).astype(np.uint8)

            bg_with_realism = bg_image.copy()
            if self.add_shadows:
                # Use advanced multi-source shadows if available
                if self.advanced_lighting_enabled and lighting_map is not None:
                    try:
                        # Create full-size object mask for shadow generation
                        full_obj_mask = np.zeros(bg_image.shape[:2], dtype=np.uint8)
                        full_obj_mask[y_start_bg:y_end_bg, x_start_bg:x_end_bg] = mask_bin_eff

                        # Object bounding box in full image coordinates
                        obj_bbox = (x_start_bg, y_start_bg, eff_w, eff_h)

                        # Estimate object height based on size (larger objects = taller)
                        obj_height_estimate = np.clip(eff_h / bg_image.shape[0], 0.05, 0.4)

                        # Apply underwater attenuation to light sources
                        adjusted_lights = apply_underwater_light_attenuation(
                            lighting_map.light_sources,
                            depth_category='mid',  # Could be determined from depth_map
                            water_clarity='clear'
                        )

                        # Generate multi-source shadows
                        shadow_mask = generate_multi_source_shadows(
                            object_mask=full_obj_mask,
                            object_bbox=obj_bbox,
                            light_sources=adjusted_lights,
                            image_size=(bg_image.shape[0], bg_image.shape[1]),
                            object_height_estimate=obj_height_estimate,
                            max_shadow_intensity=0.7
                        )

                        # Apply shadow to background
                        shadow_3ch = np.stack([shadow_mask] * 3, axis=-1)
                        bg_with_realism = bg_with_realism.astype(np.float32)
                        bg_with_realism = bg_with_realism * (1.0 - shadow_3ch)
                        bg_with_realism = np.clip(bg_with_realism, 0, 255).astype(np.uint8)

                        logging.debug(f"Applied multi-source shadows from {len(adjusted_lights)} lights")
                    except Exception as e:
                        logging.warning(f"Multi-source shadow failed: {e}. Using simple shadow.")
                        # Fallback to simple shadow
                        shadow = generate_shadow(mask_bin_eff, strength=0.6, blur_kernel_size=max(21, int(min(eff_w, eff_h) * 0.2) // 2 * 2 + 1))
                        shadow_roi = bg_with_realism[y_start_bg:y_end_bg, x_start_bg:x_end_bg]
                        s_alpha = shadow[:, :, 3].astype(np.float32) / 255.0
                        s_alpha_3ch = cv2.merge([s_alpha] * 3)
                        shadow_blended_area = shadow[:, :, :3].astype(np.float32) * s_alpha_3ch + shadow_roi.astype(np.float32) * (1 - s_alpha_3ch)
                        bg_with_realism[y_start_bg:y_end_bg, x_start_bg:x_end_bg] = np.clip(shadow_blended_area, 0, 255).astype(np.uint8)
                else:
                    # Simple shadow (original implementation)
                    shadow = generate_shadow(mask_bin_eff, strength=0.6, blur_kernel_size=max(21, int(min(eff_w, eff_h) * 0.2) // 2 * 2 + 1))
                    shadow_roi = bg_with_realism[y_start_bg:y_end_bg, x_start_bg:x_end_bg]
                    s_alpha = shadow[:, :, 3].astype(np.float32) / 255.0
                    s_alpha_3ch = cv2.merge([s_alpha] * 3)
                    shadow_blended_area = shadow[:, :, :3].astype(np.float32) * s_alpha_3ch + shadow_roi.astype(np.float32) * (1 - s_alpha_3ch)
                    bg_with_realism[y_start_bg:y_end_bg, x_start_bg:x_end_bg] = np.clip(shadow_blended_area, 0, 255).astype(np.uint8)

            if self.poisson_blending:
                center_x, center_y = x_start_bg + eff_w // 2, y_start_bg + eff_h // 2
                src_for_poisson = np.zeros_like(bg_with_realism)
                src_for_poisson[y_start_bg:y_end_bg, x_start_bg:x_end_bg] = obj_for_blending
                mask_for_poisson = np.zeros(bg_with_realism.shape[:2], dtype=np.uint8)
                mask_for_poisson[y_start_bg:y_end_bg, x_start_bg:x_end_bg] = mask_bin_eff
                final_image = apply_poisson_blending(src_for_poisson, bg_with_realism, mask_for_poisson, (center_x, center_y)) if cv2.countNonZero(mask_for_poisson) > 0 else bg_with_realism
            else:
                roi_with_shadow = bg_with_realism[y_start_bg:y_end_bg, x_start_bg:x_end_bg]

                # Multiplicación matricial precisa:
                # Pixel final = (Objeto * Alpha) + (Fondo * (1 - Alpha))
                blended_roi = (obj_for_blending.astype(np.float32) * alpha_factor_eff_3ch) + \
                              (roi_with_shadow.astype(np.float32) * (1.0 - alpha_factor_eff_3ch))
                
                final_image = bg_with_realism.copy()
                final_image[y_start_bg:y_end_bg, x_start_bg:x_end_bg] = blended_roi.astype(np.uint8)

            if final_image is not None:
                # Crear la máscara del objeto en las coordenadas de la imagen completa
                obj_mask_in_bg = np.zeros(bg_image.shape[:2], dtype=np.uint8)
                obj_mask_in_bg[y_start_bg:y_end_bg, x_start_bg:x_end_bg] = mask_bin_eff
                return final_image, candidate_box, obj_mask_in_bg, bg_with_realism

        return None, None, None, None

    def _extract_water_tint(self, background_roi):
        """Extrae el tinte dominante del agua del fondo usando análisis LAB.

        Args:
            background_roi: Región del fondo donde se pegará el objeto

        Returns:
            tuple: (B, G, R) con el color dominante del agua
        """
        try:
            # Convertir a LAB para análisis más preciso del color
            lab = cv2.cvtColor(background_roi, cv2.COLOR_BGR2LAB)

            # Analizar canales A y B (componentes de color)
            # Canal A: verde (-) a rojo (+)
            # Canal B: azul (-) a amarillo (+)
            a_mean = np.mean(lab[:, :, 1])
            b_mean = np.mean(lab[:, :, 2])

            # Extraer tonalidad dominante
            # Aguas típicamente tienen A bajo (verdoso) y B bajo (azulado)
            # Ajustar intensidad según desviación de neutro (128)
            a_shift = (a_mean - 128) * 0.3  # 30% del shift para suavizar
            b_shift = (b_mean - 128) * 0.3

            # Crear color base BGR con tendencia azul-verde típica del agua
            base_color = np.array([200, 150, 100], dtype=np.float32)  # Base azul-verde

            # Ajustar con los shifts detectados
            # A shift: -verde, +rojo
            # B shift: -azul, +amarillo
            tint = base_color.copy()
            tint[2] += a_shift  # Componente roja
            tint[1] -= a_shift * 0.5  # Componente verde (inverso parcial)
            tint[0] -= b_shift  # Componente azul (inverso)
            tint[1] += b_shift * 0.5  # Componente verde (directo parcial)

            # Clamp a rango válido
            tint = np.clip(tint, 0, 255).astype(np.uint8)

            return tuple(tint)

        except Exception as e:
            logging.warning(f"Error extrayendo water tint: {e}. Usando tinte por defecto.")
            # Tinte azul-verde por defecto para agua
            return (200, 150, 100)

    def _classify_background_context(self, bg_image):
        """Clasifica el contexto del fondo marino para matching semántico.

        Args:
            bg_image: Imagen de fondo a analizar

        Returns:
            list: Lista de contextos detectados (ej: ['open_water', 'sandy_bottom'])
        """
        try:
            # Análisis de color en espacio HSV
            hsv = cv2.cvtColor(bg_image, cv2.COLOR_BGR2HSV)
            h_mean = np.mean(hsv[:, :, 0])
            s_mean = np.mean(hsv[:, :, 1])
            v_mean = np.mean(hsv[:, :, 2])

            contexts = []

            # Agua azul abierta (H: 90-130 = azul-cian, S > 50 = saturado)
            if 90 < h_mean < 130 and s_mean > 50:
                contexts.append('open_water')

            # Arena/fondo claro (V alto = brillante, S bajo = poco saturado)
            if v_mean > 150 and s_mean < 50:
                contexts.append('sandy_bottom')

            # Rocas/fondo oscuro (V bajo = oscuro)
            if v_mean < 100:
                contexts.append('rocky_dark')

            # Tonos verdosos (algas, vegetación marina)
            if 35 < h_mean < 85 and s_mean > 30:
                contexts.append('vegetation')

            # Análisis de textura usando varianza de Laplaciano
            gray = cv2.cvtColor(bg_image, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            if laplacian_var > 500:
                contexts.append('high_texture')
            elif laplacian_var < 100:
                contexts.append('smooth')
            else:
                contexts.append('medium_texture')

            # Análisis de profundidad aparente (brillo general)
            if v_mean > 180:
                contexts.append('shallow')  # Aguas poco profundas
            elif v_mean < 80:
                contexts.append('deep')     # Aguas profundas

            # Si no se detectó ningún contexto, usar genérico
            return contexts if contexts else ['generic_underwater']

        except Exception as e:
            logging.warning(f"Error clasificando contexto de fondo: {e}")
            return ['generic_underwater']

    def _filter_backgrounds_by_context(self, bg_files, required_contexts=None):
        """Filtra fondos compatibles con contextos requeridos.

        Args:
            bg_files: Lista de paths a imágenes de fondo
            required_contexts: Lista de contextos requeridos (None = todos aceptados)

        Returns:
            list: Fondos compatibles con los contextos
        """
        if not required_contexts or not bg_files:
            return bg_files

        compatible_bgs = []

        # Cachear clasificaciones para evitar reanalizar
        if not hasattr(self, '_bg_context_cache'):
            self._bg_context_cache = {}

        for bg_file in bg_files:
            # Usar caché si existe
            if bg_file in self._bg_context_cache:
                bg_contexts = self._bg_context_cache[bg_file]
            else:
                # Clasificar fondo
                bg_image = cv2.imread(bg_file)
                if bg_image is None:
                    continue

                bg_contexts = self._classify_background_context(bg_image)
                self._bg_context_cache[bg_file] = bg_contexts

            # Verificar compatibilidad (al menos un contexto en común)
            if any(ctx in required_contexts for ctx in bg_contexts):
                compatible_bgs.append(bg_file)

        # Fallback: si no hay compatibles, devolver todos
        if not compatible_bgs:
            logging.warning(f"No se encontraron fondos compatibles con {required_contexts}. "
                          f"Usando todos los fondos disponibles.")
            return bg_files

        logging.debug(f"Filtrados {len(compatible_bgs)}/{len(bg_files)} fondos "
                     f"compatibles con {required_contexts}")

        return compatible_bgs

    def _load_and_group_objects(self, objects_source, class_mapping, selected_classes, coco_data, images_path, objects_dataset_path):
        """Carga y agrupa los objetos según la fuente y el mapeo de clases."""
        
        # Paso 1: Cargar todos los objetos de origen disponibles
        source_objects = defaultdict(list)
        if objects_source == "input_dataset" and coco_data:
            source_objects = self.extract_segmented_objects(coco_data, images_path)
        elif objects_source == "folder" and objects_dataset_path:
            all_source_folders = {src for srcs in class_mapping.values() for src in srcs}
            unmapped_classes = {cls for cls in selected_classes if cls not in class_mapping}
            folders_to_read = all_source_folders.union(unmapped_classes)
            
            logging.info(f"Leyendo carpetas de objetos para las clases de origen: {folders_to_read}")
            for source_cls in folders_to_read:
                class_dir = os.path.join(objects_dataset_path, source_cls)
                if not os.path.isdir(class_dir):
                    logging.warning(f"Carpeta de origen no encontrada, se omite: {class_dir}")
                    continue
                
                image_files = glob.glob(os.path.join(class_dir, "*.*"))
                for f in image_files:
                    img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        source_objects[source_cls].append((img, (img.shape[0], img.shape[1])))

        # Paso 2: Agrupar los objetos cargados en las clases objetivo finales
        target_objects = defaultdict(list)
        source_to_target_map = {src: target for target, srcs in class_mapping.items() for src in srcs}

        for source_name, obj_list in source_objects.items():
            target_name = source_to_target_map.get(source_name, source_name)
            target_objects[target_name].extend(obj_list)
            
        return target_objects

    def extract_segmented_objects(self, coco_data, images_path):
        """Extrae todos los objetos de un dataset COCO, sin aplicar mapeo aún."""
        source_objects = defaultdict(list)
        images_dict = {img["id"]: img for img in coco_data["images"]}
        categories = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
        
        pbar = tqdm(coco_data["annotations"], desc="Extrayendo objetos segmentados")
        for ann in pbar:
            if "segmentation" in ann and ann["segmentation"]:
                cat_name = categories.get(ann["category_id"], "")
                if not cat_name: continue
                
                image_info = images_dict.get(ann["image_id"])
                if not image_info: continue
                
                image_file = os.path.join(images_path, image_info["file_name"])
                image = cv2.imread(image_file)
                if image is None: continue

                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                for poly in ann["segmentation"]:
                    pts = np.array(poly).reshape((-1, 2)).astype(np.int32)
                    cv2.fillPoly(mask, [pts], 255)
                
                x, y, w, h = cv2.boundingRect(mask)
                if w > 0 and h > 0:
                    obj_bgr = image[y:y+h, x:x+w]
                    mask_cropped = mask[y:y+h, x:x+w]
                    obj_rgba = cv2.cvtColor(obj_bgr, cv2.COLOR_BGR2BGRA)
                    obj_rgba[:, :, 3] = mask_cropped
                    source_objects[cat_name].append((obj_rgba, (h, w)))
        return source_objects

    def validate_composite_image(self, composite_img, annotations, bg_images=None, depth_map=None):
        """
        Validate composite image quality and physics plausibility

        Args:
            composite_img: Final composite image (BGR)
            annotations: List of annotation shapes with bboxes
            bg_images: Optional list of background images for LPIPS comparison
            depth_map: Optional depth map for occlusion validation

        Returns:
            tuple: (is_valid, quality_score, violations)
        """
        if not self.validation_enabled:
            return True, None, []

        is_valid = True
        quality_score = None
        all_violations = []

        try:
            # 1. Quality validation
            if self.quality_validator:
                quality_score = self.quality_validator.validate_image(
                    composite_img,
                    reference_imgs=bg_images if bg_images else None
                )

                thresholds = self.validation_config.get('thresholds', {})

                # Check if meets minimum thresholds
                if quality_score.perceptual_quality < thresholds.get('min_perceptual_quality', 0.70):
                    is_valid = False
                    logging.debug(f"Failed perceptual quality: {quality_score.perceptual_quality:.3f}")

                if quality_score.composition_score < thresholds.get('min_composition_score', 0.70):
                    is_valid = False
                    logging.debug(f"Failed composition score: {quality_score.composition_score:.3f}")

                if quality_score.anomaly_score < thresholds.get('min_anomaly_score', 0.60):
                    is_valid = False
                    logging.debug(f"Failed anomaly score: {quality_score.anomaly_score:.3f}")

                # Detect anomalies
                anomalies = self.quality_validator.detect_anomalies(composite_img)
                if anomalies:
                    # High severity anomalies = invalid
                    critical_anomalies = [a for a in anomalies if a.severity > 0.8]
                    if critical_anomalies:
                        is_valid = False
                        logging.debug(f"Found {len(critical_anomalies)} critical anomalies")

            # 2. Physics validation
            if self.physics_validator and self.validation_config.get('metrics', {}).get('physics_checks', True):
                img_h, img_w = composite_img.shape[:2]

                # Convert annotations to COCO format for physics validator
                coco_annotations = []
                for shape in annotations:
                    if shape.get('shape_type') == 'rectangle':
                        points = shape['points']
                        x1, y1 = points[0]
                        x2, y2 = points[1]
                        bbox = [x1, y1, x2 - x1, y2 - y1]

                        coco_annotations.append({
                            'category_name': shape['label'],
                            'bbox': bbox
                        })

                # Run physics validation
                violations_dict = self.physics_validator.validate_all(
                    annotations=coco_annotations,
                    image_size=(img_w, img_h),
                    depth_map=depth_map,
                    scene_type='underwater'
                )

                # Collect all violations
                for vtype, vlist in violations_dict.items():
                    all_violations.extend(vlist)

                # Critical violations = invalid
                critical_violations = [v for v in all_violations if v.severity > 0.85]
                if critical_violations:
                    is_valid = False
                    logging.debug(f"Found {len(critical_violations)} critical physics violations")

        except Exception as e:
            logging.warning(f"Validation failed with error: {e}. Accepting image.")
            is_valid = True  # Don't reject due to validation errors

        return is_valid, quality_score, all_violations

    def augment_dataset(self, coco_data, images_path, selected_classes,
                        objects_source="dataset", objects_dataset_path=None,
                        backgrounds_dataset_path=None, max_objects_per_image=3,
                        desired_synthetic_per_class=None, class_mapping=None,
                        progress_bar=None, status_text=None, **kwargs):
        """Bucle principal para generar el dataset sintético."""
        if class_mapping is None: class_mapping = {}
        synthetic_counts_by_class = {cls: 0 for cls in selected_classes}
        
        if coco_data is None:
            required_instances = desired_synthetic_per_class.copy() if desired_synthetic_per_class else {}
        else:
            analysis = analyze_coco_dataset(coco_data, class_mapping)
            original_counts = analysis['class_counts']
            if desired_synthetic_per_class:
                required_instances = {cls: int(desired_synthetic_per_class.get(cls, 0)) for cls in selected_classes}
            else:
                target_count = max(original_counts.values()) if original_counts else 0
                required_instances = {cls: max(0, target_count - original_counts.get(cls, 0)) for cls in selected_classes}

        total_initial_required = sum(required_instances.values())
        if total_initial_required == 0:
            if status_text: status_text.info("No se requieren instancias sintéticas.")
            return {}, 0

        bg_files = glob.glob(os.path.join(backgrounds_dataset_path, "*.*"))
        if not bg_files:
            logging.error("No se encontraron fondos."); return {}, 0

        # Cargar y agrupar objetos usando la nueva lógica centralizada
        objects_by_class = self._load_and_group_objects(
            objects_source, class_mapping, selected_classes, coco_data, images_path, objects_dataset_path
        )

        for cls in selected_classes:
            if not objects_by_class.get(cls):
                logging.warning(f"No se encontraron objetos de origen para la clase objetivo '{cls}'. Se omitirá.")
                required_instances[cls] = 0

        self.category_map = {cls: idx for idx, cls in enumerate(sorted(selected_classes))}
        synthetic_total, all_annotations_data = 0, []

        # ===== CONTEXT-AWARE: Filtrar fondos compatibles con escena marina =====
        marine_contexts = ['open_water', 'sandy_bottom', 'rocky_dark', 'smooth',
                          'medium_texture', 'shallow', 'deep', 'generic_underwater']
        filtered_bg_files = self._filter_backgrounds_by_context(bg_files, marine_contexts)

        logging.info(f"Usando {len(filtered_bg_files)}/{len(bg_files)} fondos compatibles con contexto marino")

        # SAFETY BREAK for infinite loop
        loop_safety_counter = 0
        max_loop_safety = total_initial_required * 100  # Fallback generoso

        while any(required_instances.get(cls, 0) > 0 for cls in selected_classes):
            loop_safety_counter += 1
            if loop_safety_counter > max_loop_safety:
                logging.warning("Se ha excedido el límite de seguridad del bucle. Deteniendo generación para evitar bloqueo infinito.")
                break

            bg_file = random.choice(filtered_bg_files)
            bg_image = cv2.imread(bg_file)
            if bg_image is None: continue

            # ===== DEPTH MAP: Compute once per background image =====
            bg_depth_map = None
            if self.depth_aware and self.depth_estimator is not None:
                try:
                    bg_depth_map = self.depth_estimator.estimate_depth(bg_image, normalize=True)
                    logging.debug(f"Depth map computed for {bg_file}")
                except Exception as e:
                    logging.warning(f"Failed to compute depth map: {e}. Using random placement.")
                    bg_depth_map = None

            # ===== LIGHTING: Estimate once per background image =====
            bg_lighting_map = None
            if self.advanced_lighting_enabled and self.lighting_estimator is not None:
                try:
                    # Use filename as cache key
                    bg_cache_key = os.path.basename(bg_file)

                    if bg_cache_key not in self.lighting_cache:
                        bg_lighting_map = self.lighting_estimator.estimate_lighting(bg_image)
                        self.lighting_cache[bg_cache_key] = bg_lighting_map
                        logging.debug(f"Lighting estimated: {len(bg_lighting_map.light_sources)} sources detected")
                    else:
                        bg_lighting_map = self.lighting_cache[bg_cache_key]
                        logging.debug(f"Lighting retrieved from cache")
                except Exception as e:
                    logging.warning(f"Failed to estimate lighting: {e}. Using simple shadows.")
                    bg_lighting_map = None

            # Esta máscara registrará dónde se han pegado objetos.
            placement_mask = np.zeros(bg_image.shape[:2], dtype=np.uint8)

            image_base_name = f"synthetic_{synthetic_total}"
            current_bboxes, annotation_shapes = [], []
            available_classes = [cls for cls in selected_classes if required_instances.get(cls, 0) > 0 and objects_by_class.get(cls)]
            if not available_classes: break
            
            num_objetos = random.randint(1, min(max_objects_per_image, len(available_classes)))
            selected_cls_list = random.sample(available_classes, k=num_objetos)

            # ===== Z-ORDERING: Preparar objetos con información de profundidad =====
            objects_with_depth = []
            for cls in selected_cls_list:
                obj_image, original_dims = random.choice(objects_by_class[cls])
                # Calcular área del objeto (proxy de profundidad: más grande = más cerca)
                obj_area = original_dims[0] * original_dims[1] if original_dims else obj_image.shape[0] * obj_image.shape[1]
                # Añadir ruido aleatorio al área para simular profundidad variable
                depth_noise = random.uniform(0.8, 1.2)
                effective_depth = obj_area * depth_noise
                objects_with_depth.append({
                    'cls': cls,
                    'obj_image': obj_image,
                    'original_dims': original_dims,
                    'depth': effective_depth  # Mayor = más cerca (se pega después)
                })

            # Ordenar objetos por profundidad (menor primero = más lejos = se pega primero)
            objects_with_depth.sort(key=lambda x: x['depth'])

            final_image_step = bg_image.copy()

            # ===== PASTE OBJECTS IN DEPTH ORDER (back to front) =====
            for obj_idx, obj_data in enumerate(objects_with_depth):
                cls = obj_data['cls']
                obj_image = obj_data['obj_image']
                original_dims = obj_data['original_dims']

                result_image, bbox, obj_mask_in_bg, _ = self.paste_object(
                    final_image_step, obj_image.copy(), original_dims,
                    existing_bboxes=current_bboxes,
                    placement_mask=placement_mask,
                    depth_map=bg_depth_map,
                    lighting_map=bg_lighting_map
                )

                if result_image is not None and bbox is not None:
                    final_image_step = result_image
                    current_bboxes.append(bbox)

                    if obj_mask_in_bg is not None:
                        # Dilatar máscara para margen de solapamiento
                        kernel = np.ones((5,5), np.uint8)
                        dilated_mask = cv2.dilate(obj_mask_in_bg, kernel, iterations=1)
                        placement_mask = cv2.bitwise_or(placement_mask, dilated_mask)

                    # ===== CALCULAR ÁREA VISIBLE (con oclusión) =====
                    x1, y1, x2, y2 = bbox
                    # Extraer la región del objeto en la máscara de placement
                    obj_region_mask = placement_mask[y1:y2, x1:x2]
                    obj_actual_mask = obj_mask_in_bg[y1:y2, x1:x2]

                    # Calcular píxeles visibles (no ocluidos por objetos previos)
                    # Nota: placement_mask ya incluye este objeto, restar para ver oclusión
                    previous_placement = placement_mask.copy()
                    previous_placement[y1:y2, x1:x2] = cv2.bitwise_and(
                        previous_placement[y1:y2, x1:x2],
                        cv2.bitwise_not(obj_actual_mask)
                    )
                    visible_pixels = cv2.countNonZero(
                        cv2.bitwise_and(obj_actual_mask, cv2.bitwise_not(obj_region_mask))
                    ) if obj_actual_mask.shape == obj_region_mask.shape else cv2.countNonZero(obj_actual_mask)

                    total_pixels = cv2.countNonZero(obj_actual_mask)
                    occlusion_ratio = 1.0 - (visible_pixels / max(total_pixels, 1))

                    # Log oclusión si es significativa (>20%)
                    if occlusion_ratio > 0.2:
                        logging.debug(f"Objeto {cls} tiene {occlusion_ratio*100:.1f}% de oclusión")

                    annotation_shapes.append({"label": cls, "points": [[x1, y1], [x2, y2]], "shape_type": "rectangle"})
                    required_instances[cls] -= 1
                    synthetic_counts_by_class[cls] += 1
            
            if annotation_shapes:
                # Aplicar cáusticas a toda la escena sintética
                caustics = generate_caustics_map(final_image_step.shape[1], final_image_step.shape[0])
                final_image_with_caustics = apply_caustics(final_image_step, caustics)

                # Validation check
                is_valid = True
                if self.validation_enabled:
                    is_valid, quality_score, violations = self.validate_composite_image(
                        final_image_with_caustics,
                        annotation_shapes,
                        bg_images=[bg_image],
                        depth_map=bg_depth_map
                    )

                    if not is_valid:
                        if self.validation_config.get('reject_failed_images', True):
                            logging.info(f"Image rejected due to validation failure. Score: {quality_score}")

                            # Save rejected image for debugging if enabled
                            if self.validation_config.get('save_rejected', True):
                                rejected_name = f"rejected_{synthetic_total}_{loop_safety_counter}.jpg"
                                rejected_path = os.path.join(self.rejected_dir, rejected_name)
                                cv2.imwrite(rejected_path, final_image_with_caustics)

                                # Save validation info
                                validation_info = {
                                    'quality_score': str(quality_score) if quality_score else None,
                                    'violations': [str(v) for v in violations],
                                    'timestamp': loop_safety_counter
                                }
                                info_path = os.path.join(self.rejected_dir, f"rejected_{synthetic_total}_{loop_safety_counter}.json")
                                with open(info_path, 'w') as f:
                                    json.dump(validation_info, f, indent=2)

                            self.rejected_images.append({
                                'iteration': loop_safety_counter,
                                'quality_score': quality_score,
                                'violations': violations
                            })

                            # Skip saving this image - continue to next iteration
                            continue

                # Image passed validation - save it
                if is_valid:
                    image_name = f"{image_base_name}.jpg"
                    output_image_path = os.path.join(self.images_dir, image_name)
                    cv2.imwrite(output_image_path, final_image_with_caustics)
                    annotated_image = self.draw_annotations_on_image(final_image_step, annotation_shapes)
                    cv2.imwrite(os.path.join(self.annotated_dir, image_name), annotated_image)
                    all_annotations_data.append({"imagePath": image_name, "imageHeight": bg_image.shape[0], "imageWidth": bg_image.shape[1], "shapes": annotation_shapes})
                    synthetic_total += 1

            total_remaining = sum(required_instances.values())
            progress = 1.0 - (total_remaining / total_initial_required)
            if progress_bar: progress_bar.progress(progress)
            if status_text: status_text.text(f"Progreso: {int(progress*100)}% | Imágenes generadas: {synthetic_total}")

        self.generate_coco_json(all_annotations_data)
        return synthetic_counts_by_class, synthetic_total

    def draw_annotations_on_image(self, image, shapes):
        annotated = image.copy()
        unique_labels = sorted(list(set(shape['label'] for shape in shapes)))
        color_map = {label: COLOR_PALETTE[i % len(COLOR_PALETTE)] for i, label in enumerate(unique_labels)}
        for shape in shapes:
            label, points, shape_type = shape["label"], shape["points"], shape["shape_type"]
            color = color_map.get(label, (255, 255, 255))
            if shape_type == "rectangle" and len(points) == 2:
                p1, p2 = (int(points[0][0]), int(points[0][1])), (int(points[1][0]), int(points[1][1]))
                cv2.rectangle(annotated, p1, p2, color, thickness=2)
                cv2.putText(annotated, label, (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return annotated

    def generate_coco_json(self, all_annotations_data):
        all_images_info, all_annotations_info, ann_id_counter = [], [], 1
        for img_id, ann_data in enumerate(all_annotations_data):
            all_images_info.append({"id": img_id, "file_name": ann_data["imagePath"], "width": ann_data["imageWidth"], "height": ann_data["imageHeight"]})
            for shape in ann_data["shapes"]:
                x1, y1 = shape["points"][0]; x2, y2 = shape["points"][1]
                bbox = [x1, y1, x2 - x1, y2 - y1]
                all_annotations_info.append({"id": ann_id_counter, "image_id": img_id, "category_id": self.category_map.get(shape["label"], -1), "segmentation": [], "area": bbox[2] * bbox[3], "bbox": bbox, "iscrowd": 0})
                ann_id_counter += 1
        categories_list = [{"id": cat_id, "name": name, "supercategory": "none"} for name, cat_id in self.category_map.items()]
        coco_synthetic = {"info": {"description": "Synthetic Dataset with Realism"}, "licenses": [], "images": all_images_info, "annotations": all_annotations_info, "categories": categories_list}
        output_json_path = os.path.join(self.output_dir, "synthetic_dataset.json")
        with open(output_json_path, "w") as f:
            json.dump(coco_synthetic, f, indent=2)
        logging.info(f"JSON COCO sintético guardado en {output_json_path}")