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
from src.augmentation.transformations import rotate_image, scale_image, apply_perspective_transform, apply_motion_blur
from src.augmentation.realism import (apply_poisson_blending, transfer_color_correction,
                                      match_blur, generate_shadow, add_lighting_effect,
                                      apply_underwater_effect, add_upscaling_noise, generate_caustics_map, apply_caustics)

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
        """
        Comprueba de forma segura el solapamiento entre la máscara de un candidato y la de ocupación,
        manejando correctamente los casos en los bordes de la imagen.
        """
        x, y = candidate_pos
        cand_h, cand_w = candidate_mask.shape
        bg_h, bg_w = placement_mask.shape
        
        # 1. Calcular las coordenadas de la región de superposición (ROI) en la imagen de fondo.
        #    Esto asegura que no nos salgamos de los límites.
        roi_x_start = max(0, x)
        roi_y_start = max(0, y)
        roi_x_end = min(bg_w, x + cand_w)
        roi_y_end = min(bg_h, y + cand_h)

        # 2. Si la región de superposición no tiene área, es imposible que se solapen.
        if roi_x_start >= roi_x_end or roi_y_start >= roi_y_end:
            return False

        # 3. Calcular las coordenadas correspondientes para recortar la máscara del candidato.
        #    Esto es crucial para que ambas máscaras tengan el mismo tamaño.
        cand_roi_x_start = roi_x_start - x
        cand_roi_y_start = roi_y_start - y
        cand_roi_x_end = cand_roi_x_start + (roi_x_end - roi_x_start)
        cand_roi_y_end = cand_roi_y_start + (roi_y_end - roi_y_start)
        
        # 4. Recortar ambas máscaras para que tengan exactamente el mismo tamaño.
        roi_placement = placement_mask[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        roi_candidate = candidate_mask[cand_roi_y_start:cand_roi_y_end, cand_roi_x_start:cand_roi_x_end]

        # Ahora 'roi_placement' y 'roi_candidate' tienen garantizado el mismo tamaño.
        intersection = cv2.bitwise_and(roi_placement, roi_candidate)
        
        candidate_area = cv2.countNonZero(roi_candidate)
        if candidate_area == 0:
            return False  # El objeto no tiene área visible en esta ROI
            
        overlap_area = cv2.countNonZero(intersection)
        overlap_ratio = overlap_area / candidate_area
        
        return overlap_ratio > threshold

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
                 max_area_ratio=0.4):
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
    
    def paste_object(self, bg_image, patch, original_dims, existing_bboxes=None, placement_mask=None):
        """Intenta pegar un objeto en el fondo aplicando realismo y controlando solapamiento."""
        if existing_bboxes is None: existing_bboxes = []
        if placement_mask is None: placement_mask = np.zeros(bg_image.shape[:2], dtype=np.uint8)
        bg_h, bg_w = bg_image.shape[:2]
        bg_area = bg_h * bg_w

        for i in range(self.try_count):
            transformed_patch, pos = self.apply_transformations(patch, bg_image.shape, original_dims)

            if transformed_patch is None or transformed_patch.shape[0] == 0 or transformed_patch.shape[1] == 0:
                continue

            if transformed_patch.shape[2] < 4:
                alpha_channel = np.ones(transformed_patch.shape[:2], dtype=np.uint8) * 255
                obj_bgr = transformed_patch
            else:
                alpha_channel = transformed_patch[:, :, 3]
                obj_bgr = transformed_patch[:, :, :3]

            _, mask_bin = cv2.threshold(alpha_channel, 127, 255, cv2.THRESH_BINARY)

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
            alpha_channel_eff = alpha_channel[y_start_patch:y_start_patch+eff_h, x_start_patch:x_start_patch+eff_w]
            roi = bg_image[y_start_bg:y_end_bg, x_start_bg:x_end_bg]

            obj_for_blending = obj_bgr_eff.copy()
            if self.underwater_effect: obj_for_blending = apply_underwater_effect(obj_for_blending, intensity=self.realism_intensity)
            if self.advanced_color_correction: obj_for_blending = transfer_color_correction(obj_for_blending, roi, mask_bin_eff, intensity=self.realism_intensity)
            if self.blur_consistency: obj_for_blending = match_blur(obj_for_blending, roi, mask_bin_eff)
            if self.lighting_effects: obj_for_blending = add_lighting_effect(obj_for_blending)
            if self.motion_blur: obj_for_blending = apply_motion_blur(obj_for_blending)

            bg_with_realism = bg_image.copy()
            if self.add_shadows:
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
                #blur_size = int(min(eff_w, eff_h) * 0.05) // 2 * 2 + 1
                blur_factor = 0.02
                max_blur_size = 7 
                blur_size = int(min(eff_w, eff_h) * blur_factor) // 2 * 2 + 1
                blur_size = min(max_blur_size, max(3, blur_size))

                blurred_alpha = cv2.GaussianBlur(alpha_channel_eff, (blur_size, blur_size), 0)
                alpha_norm = (blurred_alpha.astype(np.float32) / 255.0)[:, :, np.newaxis]
                roi_with_shadow = bg_with_realism[y_start_bg:y_end_bg, x_start_bg:x_end_bg]
                blended_roi = obj_for_blending * alpha_norm + roi_with_shadow * (1 - alpha_norm)
                final_image = bg_with_realism.copy()
                final_image[y_start_bg:y_end_bg, x_start_bg:x_end_bg] = blended_roi.astype(np.uint8)

            if final_image is not None:
                # Crear la máscara del objeto en las coordenadas de la imagen completa
                obj_mask_in_bg = np.zeros(bg_image.shape[:2], dtype=np.uint8)
                obj_mask_in_bg[y_start_bg:y_end_bg, x_start_bg:x_end_bg] = mask_bin_eff
                return final_image, candidate_box, obj_mask_in_bg, bg_with_realism

        return None, None, None, None

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

        while any(required_instances.get(cls, 0) > 0 for cls in selected_classes):
            bg_file = random.choice(bg_files)
            bg_image = cv2.imread(bg_file)
            if bg_image is None: continue

            # Esta máscara registrará dónde se han pegado objetos.
            placement_mask = np.zeros(bg_image.shape[:2], dtype=np.uint8)

            image_base_name = f"synthetic_{synthetic_total}"
            current_bboxes, annotation_shapes = [], []
            available_classes = [cls for cls in selected_classes if required_instances.get(cls, 0) > 0 and objects_by_class.get(cls)]
            if not available_classes: break
            
            num_objetos = random.randint(1, min(max_objects_per_image, len(available_classes)))
            selected_cls_list = random.sample(available_classes, k=num_objetos)
            final_image_step = bg_image.copy()
            
            for obj_idx, cls in enumerate(selected_cls_list):
                obj_image, original_dims = random.choice(objects_by_class[cls])
                result_image, bbox, obj_mask_in_bg, _ = self.paste_object(
                    final_image_step, obj_image.copy(), original_dims, 
                    existing_bboxes=current_bboxes, 
                    placement_mask=placement_mask # Pasar la máscara
                )
                
                if result_image is not None and bbox is not None:
                    final_image_step = result_image
                    current_bboxes.append(bbox)

                    if obj_mask_in_bg is not None:
                        # Dilatar un poco la máscara para crear un margen y evitar solapamientos en los bordes
                        kernel = np.ones((5,5), np.uint8)
                        dilated_mask = cv2.dilate(obj_mask_in_bg, kernel, iterations=1)
                        placement_mask = cv2.bitwise_or(placement_mask, dilated_mask)

                    x1, y1, x2, y2 = bbox
                    annotation_shapes.append({"label": cls, "points": [[x1, y1], [x2, y2]], "shape_type": "rectangle"})
                    required_instances[cls] -= 1
                    synthetic_counts_by_class[cls] += 1
            
            if annotation_shapes:
                # Aplicar cáusticas a toda la escena sintética
                caustics = generate_caustics_map(final_image_step.shape[1], final_image_step.shape[0])
                final_image_with_caustics = apply_caustics(final_image_step, caustics)
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