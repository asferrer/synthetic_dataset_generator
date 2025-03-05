# src/augmentation/augmentor.py

import os
import json
import random
import math
import base64
import cv2
import numpy as np
import imageio
from scipy import ndimage
from tqdm import tqdm
import logging
import glob

from src.utils.helpers import ensure_dir
from src.augmentation.transformations import rotate_image, scale_image, translate_image

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(levelname)s: %(message)s')

def compute_iou(boxA, boxB):
    # boxA and boxB are (x1, y1, x2, y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    unionArea = boxAArea + boxBArea - interArea
    return interArea / unionArea if unionArea != 0 else 0

class SyntheticDataAugmentor:
    def __init__(self,
                 output_dir="/app/synthetic_dataset",
                 rot=True,
                 scale=True,
                 trans=True,
                 try_count=3,
                 overlap_threshold=10,
                 seed=1):
        self.rot = rot
        self.scale = scale
        self.trans = trans
        self.try_count = try_count
        self.overlap_threshold = overlap_threshold
        self.output_dir = output_dir
        random.seed(seed)
        
        self.images_dir = os.path.join(self.output_dir, "images")
        self.labels_dir = os.path.join(self.output_dir, "labels")
        ensure_dir(self.images_dir)
        ensure_dir(self.labels_dir)
        logging.debug(f"Directorios creados: {self.images_dir}, {self.labels_dir}")

    def rotate_point(self, point, angle, origin):
        x, y = point
        ox, oy = origin
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        x_adj = x - ox
        y_adj = y - oy
        x_new = ox + cos_a * x_adj - sin_a * y_adj
        y_new = oy + sin_a * x_adj + cos_a * y_adj
        return [x_new, y_new]

    def apply_transformations(self, patch, bg_shape):
        transformed_patch = patch.copy()
        angle = 0
        scale_factor = 1.0
        h_bg, w_bg = bg_shape[:2]
        
        if self.rot:
            angle = random.randrange(0, 360)
            transformed_patch = rotate_image(transformed_patch, angle)
            logging.debug(f"Patch rotado a {angle}°")
        
        p_h, p_w = transformed_patch.shape[:2]
        if self.scale:
            max_scale = min(h_bg / p_h, w_bg / p_w)
            scale_factor = random.uniform(0.25, max(0.25, 0.8 * max_scale))
            transformed_patch = scale_image(transformed_patch, scale_factor)
            logging.debug(f"Patch escalado con factor {scale_factor:.2f}")
        
        p_h, p_w = transformed_patch.shape[:2]
        if self.trans:
            max_x = max(w_bg - p_w, 1)
            max_y = max(h_bg - p_h, 1)
            x = random.randrange(0, max_x)
            y = random.randrange(0, max_y)
        else:
            x, y = 0, 0
        pos = (x, y)
        logging.debug(f"Patch trasladado a la posición {pos}")
        return transformed_patch, angle, scale_factor, pos

    def _paste_png(self, bg, patch, pos):
        x, y = pos
        hp, wp = patch.shape[:2]
        bg_h, bg_w = bg.shape[:2]

        new_hp, new_wp = hp, wp
        if y + hp > bg_h:
            new_hp = bg_h - y
            patch = patch[:new_hp, :, :]
        if x + wp > bg_w:
            new_wp = bg_w - x
            patch = patch[:, :new_wp, :]

        if patch.shape[2] < 4:
            mask = 255 * np.ones((new_hp, new_wp), dtype=np.uint8)
            patch_rgba = cv2.cvtColor(patch, cv2.COLOR_BGR2BGRA)
            patch_rgba[:, :, 3] = mask
            patch = patch_rgba

        # Separar canales
        b, g, r, a = cv2.split(patch)

        # --- Adaptación de tono ---
        roi = bg[y:y+new_hp, x:x+new_wp]
        bg_mean = cv2.mean(roi)[:3]
        alpha_norm = a.astype(np.float32) / 255.0
        obj_mean = []
        for channel in (b, g, r):
            channel_mean = np.sum(channel.astype(np.float32) * alpha_norm) / (np.sum(alpha_norm) + 1e-8)
            obj_mean.append(channel_mean)
        obj_mean = np.array(obj_mean)
        bg_mean = np.array(bg_mean)
        diff = bg_mean - obj_mean
        b = np.clip(b.astype(np.float32) + diff[0], 0, 255).astype(np.uint8)
        g = np.clip(g.astype(np.float32) + diff[1], 0, 255).astype(np.uint8)
        r = np.clip(r.astype(np.float32) + diff[2], 0, 255).astype(np.uint8)
        patch_bgr = cv2.merge((b, g, r))
        # --- Suavizado de bordes ---
        blurred_alpha = cv2.GaussianBlur(a, (7, 7), 0)
        blurred_alpha_norm = blurred_alpha.astype(np.float32) / 255.0
        # --- Mezcla ---
        roi_float = roi.astype(np.float32)
        patch_float = patch_bgr.astype(np.float32)
        alpha_3ch = cv2.merge([blurred_alpha_norm, blurred_alpha_norm, blurred_alpha_norm])
        blended = patch_float * alpha_3ch + roi_float * (1 - alpha_3ch)
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        result = bg.copy()
        result[y:y+new_hp, x:x+new_wp] = blended
        logging.debug(f"Patch pegado en {pos} con dimensiones efectivas {(new_hp, new_wp)}")
        return result, new_hp, new_wp

    def _paste_bin_png(self, bg, patch, idx_obj, pos):
        x, y = pos
        p_h, p_w = patch.shape[:2]
        bg_h, bg_w = bg.shape[:2]
        
        if y + p_h > bg_h:
            p_h = bg_h - y
            patch = patch[:p_h, :, :]
        if x + p_w > bg_w:
            p_w = bg_w - x
            patch = patch[:, :p_w, :]

        patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        _, patch_bw = cv2.threshold(patch_gray, 0, 1, cv2.THRESH_BINARY)
        patch_bw = patch_bw * idx_obj
        patch_bgr = cv2.cvtColor(patch_bw.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        mask_full = np.zeros_like(bg, dtype=np.uint8)
        mask_full[y:y+p_h, x:x+p_w] = patch_bgr
        _, mask_full_bin = cv2.threshold(mask_full, 0, 255, cv2.THRESH_BINARY)
        
        bg_copy = bg.copy()
        object_areas = {}
        unique_ids = np.unique(bg_copy)
        for obj_id in unique_ids:
            if obj_id == 0:
                continue
            area_existing = np.count_nonzero(bg_copy == obj_id) / 3.0
            mask_obj = (bg_copy == obj_id).astype(np.uint8) * 255
            overlapping = cv2.bitwise_and(mask_obj, mask_full_bin)
            overlapping_area = np.count_nonzero(overlapping) / 3.0
            if area_existing > 0:
                object_areas[obj_id] = 100 * overlapping_area / area_existing
        bg_updated = cv2.add(bg_copy, mask_full)
        _, bg_updated = cv2.threshold(bg_updated, idx_obj, idx_obj, cv2.THRESH_TRUNC)
        logging.debug(f"Pasting bin: idx_obj={idx_obj}, overlapping areas: {object_areas}")
        return bg_updated, object_areas

    def paste_object(self, bg_bin, bg_image, patch, idx_obj, existing_mask=None, max_overlap_threshold=0.1):
        """
        Intenta pegar un objeto sobre el fondo, controlando el solapamiento mediante la máscara del objeto.
        Se utiliza una dilatación en la máscara del candidato para asegurar que se detecten intersecciones pequeñas.
        
        Parámetros:
        - bg_bin: la máscara acumulada de objetos ya pegados (imagen binaria).
        - bg_image: imagen de fondo.
        - patch: el objeto a pegar (con canal alfa, o se crea uno si no existe).
        - idx_obj: índice del objeto para la operación.
        - existing_mask: máscara acumulada de objetos ya pegados (del mismo tamaño que bg_image).
        - max_overlap_threshold: fracción máxima de solapamiento permitida (por defecto 0.1).
        
        Retorna:
        - bg_bin: máscara actualizada.
        - bg_image: imagen de fondo actualizada con el objeto pegado.
        - final_angle, final_pos, final_scale: parámetros de transformación aplicados.
        - patched: booleano que indica si el objeto fue pegado.
        - effective_hp, effective_wp: dimensiones efectivas del área pegada.
        - candidate_box: bounding box (x1, y1, x2, y2) del área pegada.
        - updated_mask: la máscara acumulada actualizada.
        """
        patched = False
        final_angle, final_pos, final_scale = 0, (0, 0), 1.0
        effective_hp, effective_wp = 0, 0
        candidate_box = None
        updated_mask = existing_mask.copy() if existing_mask is not None else np.zeros(bg_image.shape[:2], dtype=np.uint8)
        
        # Kernel para dilatación (para ampliar los bordes)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        for i in range(self.try_count):
            transformed_patch, angle, scale_factor, pos = self.apply_transformations(patch, bg_image.shape)
            bg_bin_candidate, _ = self._paste_bin_png(bg_bin, transformed_patch, idx_obj, pos)
            
            # Extraer la máscara candidata del patch (si tiene canal alfa, se usa; si no, se asume máscara llena)
            if transformed_patch.shape[2] < 4:
                candidate_mask = np.ones(transformed_patch.shape[:2], dtype=np.uint8) * 255
            else:
                candidate_mask = transformed_patch[:, :, 3]
            _, candidate_mask_bin = cv2.threshold(candidate_mask, 127, 255, cv2.THRESH_BINARY)
            
            # Obtener las dimensiones efectivas del candidato considerando los límites del fondo
            bg_h, bg_w = bg_image.shape[:2]
            candidate_h = min(candidate_mask_bin.shape[0], bg_h - pos[1])
            candidate_w = min(candidate_mask_bin.shape[1], bg_w - pos[0])
            candidate_mask_resized = candidate_mask_bin[:candidate_h, :candidate_w]
            
            # Crear una máscara a escala del fondo (inicialmente ceros) e insertar el candidato en la posición
            candidate_mask_pasted = np.zeros(bg_image.shape[:2], dtype=np.uint8)
            candidate_mask_pasted[pos[1]:pos[1]+candidate_h, pos[0]:pos[0]+candidate_w] = candidate_mask_resized
            
            # Dilatar la máscara candidata para "ampliar" el contorno
            dilated_candidate = cv2.dilate(candidate_mask_pasted, kernel, iterations=1)
            
            # Calcular el solapamiento con la máscara acumulada (existing_mask)
            if existing_mask is not None:
                overlap_pixels = np.sum((existing_mask > 0) & (dilated_candidate > 0))
            else:
                overlap_pixels = 0
            candidate_area = np.sum(dilated_candidate > 0)
            overlap_ratio = overlap_pixels / candidate_area if candidate_area > 0 else 0
            
            logging.debug(f"Intento {i+1}: Overlap ratio = {overlap_ratio:.2f}")
            if overlap_ratio > max_overlap_threshold:
                logging.debug("Candidato rechazado por solapamiento excesivo.")
                continue  # Reintentar sin pegar
            
            # Si el candidato es aceptable, pegar el objeto usando _paste_png
            candidate_bg_image, effective_hp, effective_wp = self._paste_png(bg_image, transformed_patch, pos)
            candidate_box = (pos[0], pos[1], pos[0] + effective_wp, pos[1] + effective_hp)
            # Actualizar la máscara acumulada con el candidato (sin dilatación para conservar el área real)
            updated_mask = cv2.bitwise_or(updated_mask, candidate_mask_pasted)
            
            # Aceptar el candidato
            bg_bin = bg_bin_candidate
            bg_image = candidate_bg_image
            final_angle, final_pos, final_scale = angle, pos, scale_factor
            patched = True
            break

        if not patched:
            logging.warning("No se pudo pegar el objeto sin solapamiento excesivo.")
        return bg_bin, bg_image, final_angle, final_pos, final_scale, patched, effective_hp, effective_wp, candidate_box, updated_mask



    def update_annotation(self, image_name, json_path, angle, pos, scale_factor, patch, new_image, image_full_path):
        try:
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                logging.debug(f"JSON original cargado desde {json_path}")
            else:
                data = {
                    "version": "4.5.7",
                    "flags": {},
                    "shapes": [],
                    "imagePath": image_name,
                    "imageData": None,
                    "imageHeight": new_image.shape[0],
                    "imageWidth": new_image.shape[1]
                }
                logging.debug("No se encontró JSON original; se crea estructura base.")
        except Exception as e:
            logging.error(f"Error al abrir {json_path}: {e}")
            return

        data["imageWidth"] = new_image.shape[1]
        data["imageHeight"] = new_image.shape[0]
        data["imagePath"] = image_name

        try:
            with open(image_full_path, "rb") as img_file:
                encoded = base64.b64encode(img_file.read()).decode("utf-8")
            data["imageData"] = encoded
            logging.debug("Imagen codificada en base64 correctamente.")
        except Exception as e:
            logging.error(f"Error al codificar la imagen {image_full_path}: {e}")

        x, y = pos
        p_h, p_w = patch.shape[:2]
        new_shape = {
            "label": "augmented_object",
            "points": [[x, y], [x + p_w, y], [x + p_w, y + p_h], [x, y + p_h]],
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }
        data["shapes"].append(new_shape)
        logging.debug(f"Se añadió anotación: {new_shape}")

        json_name = image_name.replace(".jpg", ".json")
        dest_json_path = os.path.join(self.labels_dir, json_name)
        try:
            with open(dest_json_path, "w") as f:
                json.dump(data, f, indent=2, separators=(", ", ": "))
            logging.debug(f"JSON guardado en {dest_json_path}")
        except Exception as e:
            logging.error(f"Error al escribir {dest_json_path}: {e}")

    def draw_annotations_on_image(self, image, shapes):
        annotated = image.copy()
        for shape in shapes:
            label = shape.get("label", "unknown")
            points = shape.get("points", [])
            if len(points) >= 2:
                top_left = (int(points[0][0]), int(points[0][1]))
                bottom_right = (int(points[2][0]), int(points[2][1]))
                cv2.rectangle(annotated, top_left, bottom_right, (0, 255, 0), 2)
                cv2.putText(annotated, label, (top_left[0], top_left[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return annotated

    def extract_segmented_objects(self, coco_data, images_path, selected_classes):
        objects_by_class = {cls: [] for cls in selected_classes}
        images_dict = {img["id"]: img for img in coco_data["images"]}
        categories = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
        
        for ann in coco_data["annotations"]:
            if "segmentation" in ann and ann["segmentation"]:
                cat_name = categories.get(ann["category_id"], "")
                if cat_name in selected_classes:
                    image_info = images_dict.get(ann["image_id"])
                    if not image_info:
                        continue
                    image_file = os.path.join(images_path, image_info["file_name"])
                    image = cv2.imread(image_file)
                    if image is None:
                        continue
                    seg = ann["segmentation"]
                    if isinstance(seg, list):
                        mask = np.zeros(image.shape[:2], dtype=np.uint8)
                        for poly in seg:
                            pts = np.array(poly).reshape((-1, 2)).astype(np.int32)
                            cv2.fillPoly(mask, [pts], 255)
                        obj = cv2.bitwise_and(image, image, mask=mask)
                        x, y, w, h = cv2.boundingRect(mask)
                        if w > 0 and h > 0:
                            cropped = obj[y:y+h, x:x+w]
                            objects_by_class[cat_name].append(cropped)
                    else:
                        logging.warning("Formato de segmentación no soportado; se ignora.")
        return objects_by_class

    def augment_dataset(self, coco_data, images_path, selected_classes,
                        objects_source="dataset", objects_dataset_path=None,
                        backgrounds_dataset_path=None, max_objects_per_image=3,
                        desired_synthetic_per_class=None, progress_bar=None):
        """
        Genera imágenes sintéticas para aumentar el dataset para las clases seleccionadas,
        hasta alcanzar el número deseado de muestras por clase (definido en desired_synthetic_per_class).
        
        Parámetros:
        - coco_data: Diccionario del dataset COCO de entrada.
        - images_path: Ruta de las imágenes originales.
        - selected_classes: Lista de nombres de clases a aumentar.
        - objects_source: "dataset" para extraer objetos segmentados del dataset o "folder" para usar objetos externos.
        - objects_dataset_path: Ruta a la carpeta con objetos (si se usa "folder").
        - backgrounds_dataset_path: Ruta a la carpeta con fondos.
        - max_objects_per_image: Número máximo de objetos a pegar en cada imagen sintética.
        - desired_synthetic_per_class: (Opcional) Diccionario con el número deseado de muestras sintéticas por clase.
                                        Si se especifica, se generarán exactamente ese número para cada clase.
        - progress_bar: (Opcional) Objeto de progreso de Streamlit.
        
        Retorna:
        - synthetic_counts_by_class: Diccionario con la cantidad de instancias sintéticas generadas por cada clase.
        - synthetic_total: Total de imágenes sintéticas generadas.
        """
        # Inicializar el contador de muestras sintéticas por clase
        synthetic_counts_by_class = {cls: 0 for cls in selected_classes}
        synthetic_count = 0

        # Calcular conteos originales de las clases seleccionadas
        categories = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
        original_counts = {}
        for ann in coco_data["annotations"]:
            cat_name = categories.get(ann["category_id"], "")
            if cat_name in selected_classes:
                original_counts[cat_name] = original_counts.get(cat_name, 0) + 1
        if not original_counts:
            logging.warning("No se encontraron instancias originales para las clases seleccionadas.")
            return synthetic_counts_by_class, synthetic_count

        # Definir el número de instancias a generar por clase:
        # Si se especifica desired_synthetic_per_class, se usa ese valor;
        # de lo contrario, se balancea el dataset al nivel de la clase con mayor instancias.
        if desired_synthetic_per_class is not None:
            required_instances = {cls: int(desired_synthetic_per_class.get(cls, 0))
                                for cls in selected_classes}
        else:
            target_count = max(original_counts.values())
            required_instances = {cls: max(target_count - original_counts.get(cls, 0), 0)
                                for cls in selected_classes}
        initial_required = sum(required_instances.values())
        logging.info(f"Instancias sintéticas requeridas: {required_instances}")

        # Generar el mapeo de categorías para el JSON global
        self.category_map = {}
        for cat in coco_data["categories"]:
            if cat["name"] in selected_classes:
                self.category_map[cat["name"]] = cat["id"]
        next_cat_id = 1
        for cls in selected_classes:
            if cls not in self.category_map:
                self.category_map[cls] = next_cat_id
                next_cat_id += 1

        # Acumuladores para la generación del JSON global
        synthetic_images_info = []
        synthetic_annotations = []
        ann_id = 1

        # Función auxiliar para actualizar la barra de progreso
        def update_progress():
            if progress_bar is not None and initial_required > 0:
                current_required = sum(required_instances.values())
                progress = (initial_required - current_required) / initial_required
                progress_bar.progress(min(progress, 1.0))

        # Rama para cuando se utilizan objetos extraídos del dataset (modo segmentación)
        if objects_source == "dataset":
            logging.info("Modo 'dataset' activado: utilizando objetos segmentados del dataset de entrada.")
            segmentation_mode = any("segmentation" in ann and ann["segmentation"] for ann in coco_data["annotations"])
            if not segmentation_mode:
                logging.error("No se detectaron anotaciones de segmentación en el dataset.")
                return synthetic_counts_by_class, synthetic_count
            objects_by_class = self.extract_segmented_objects(coco_data, images_path, selected_classes)
            if not backgrounds_dataset_path:
                logging.error("Se requiere un directorio de fondos para el modo 'dataset'.")
                return synthetic_counts_by_class, synthetic_count
            bg_files = [f for f in glob.glob(os.path.join(backgrounds_dataset_path, "*.*"))
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not bg_files:
                logging.error("No se encontraron fondos en el directorio especificado.")
                return synthetic_counts_by_class, synthetic_count
            annotated_dir = os.path.join(self.output_dir, "annotated")
            ensure_dir(annotated_dir)
            
            while sum(required_instances.values()) > 0:
                num_objetos = random.randint(1, min(max_objects_per_image, sum(required_instances.values())))
                logging.debug(f"Generando imagen sintética con {num_objetos} objetos.")
                available_classes = [cls for cls, req in required_instances.items() if req > 0]
                if not available_classes:
                    break
                if num_objetos > len(available_classes):
                    selected_cls_list = random.choices(available_classes, k=num_objetos)
                else:
                    selected_cls_list = random.sample(available_classes, k=num_objetos)
                bg_file = random.choice(bg_files)
                bg_image = cv2.imread(bg_file)
                if bg_image is None:
                    logging.error(f"Error al cargar fondo {bg_file}")
                    continue
                annotation_shapes = []
                bg_bin = np.zeros_like(bg_image, dtype=np.uint8)
                existing_mask = np.zeros(bg_image.shape[:2], dtype=np.uint8)
                for cls in selected_cls_list:
                    seg_objects = objects_by_class.get(cls, [])
                    if not seg_objects:
                        logging.warning(f"No se encontraron objetos segmentados para {cls}.")
                        continue
                    obj_image = random.choice(seg_objects)
                    result = self.paste_object(bg_bin, bg_image, obj_image, idx_obj=1,
                                            existing_mask=existing_mask, max_overlap_threshold=0.2)
                    (bg_bin, bg_image, angle, pos, scale_factor, patched,
                    eff_hp, eff_wp, candidate_box, updated_mask) = result
                    if patched and candidate_box is not None:
                        existing_mask = updated_mask
                        x, y, x2, y2 = candidate_box
                        shape = {
                            "label": cls,
                            "points": [[x, y], [x2, y], [x2, y2], [x, y2]],
                            "group_id": None,
                            "shape_type": "rectangle",
                            "flags": {}
                        }
                        annotation_shapes.append(shape)
                        required_instances[cls] = max(required_instances[cls] - 1, 0)
                        synthetic_counts_by_class[cls] += 1
                    else:
                        logging.warning(f"No se pudo pegar objeto para {cls}.")
                if annotation_shapes:
                    image_name = f"synthetic_{synthetic_count}.jpg"
                    output_image_path = os.path.join(self.images_dir, image_name)
                    cv2.imwrite(output_image_path, bg_image)
                    logging.info(f"Imagen sintética guardada: {output_image_path}")
                    annotation = {
                        "version": "4.5.7",
                        "flags": {},
                        "shapes": annotation_shapes,
                        "imagePath": image_name,
                        "imageData": None,
                        "imageHeight": bg_image.shape[0],
                        "imageWidth": bg_image.shape[1]
                    }
                    try:
                        with open(output_image_path, "rb") as img_file:
                            encoded = base64.b64encode(img_file.read()).decode("utf-8")
                        annotation["imageData"] = encoded
                    except Exception as e:
                        logging.error(f"Error al codificar {output_image_path}: {e}")
                    json_name = image_name.replace(".jpg", ".json")
                    dest_json_path = os.path.join(self.labels_dir, json_name)
                    try:
                        with open(dest_json_path, "w") as f:
                            json.dump(annotation, f, indent=2, separators=(", ", ": "))
                        logging.info(f"Anotación guardada: {dest_json_path}")
                    except Exception as e:
                        logging.error(f"Error al guardar anotación en {dest_json_path}: {e}")
                    synthetic_count += 1
                    annotated_image = self.draw_annotations_on_image(bg_image, annotation_shapes)
                    annotated_image_path = os.path.join(annotated_dir, image_name)
                    cv2.imwrite(annotated_image_path, annotated_image)
                    logging.info(f"Imagen anotada guardada: {annotated_image_path}")
                    synthetic_images_info.append({
                        "id": synthetic_count,
                        "file_name": image_name,
                        "width": bg_image.shape[1],
                        "height": bg_image.shape[0]
                    })
                    for shape in annotation_shapes:
                        x1, y1 = shape["points"][0]
                        x2, y2 = shape["points"][2]
                        bbox = [x1, y1, x2 - x1, y2 - y1]
                        synthetic_annotations.append({
                            "id": ann_id,
                            "image_id": synthetic_count,
                            "category_id": self.category_map.get(shape["label"], 0),
                            "segmentation": [],
                            "area": bbox[2] * bbox[3],
                            "bbox": bbox,
                            "iscrowd": 0
                        })
                        ann_id += 1
                else:
                    logging.debug("No se pegaron objetos en esta iteración.")
                update_progress()
            logging.info("Proceso de augmentación en modo 'dataset' completado.")
        
        # Rama para cuando se utiliza objetos desde carpeta externa (modo "folder")
        elif objects_source == "folder" and objects_dataset_path and backgrounds_dataset_path:
            logging.info("Modo 'folder' activado: utilizando objetos de carpeta y fondos.")
            # Construir diccionario de objetos por clase desde la carpeta externa.
            objects_by_class = {}
            for cls in selected_classes:
                class_dir = os.path.join(objects_dataset_path, cls)
                image_files = [f for f in glob.glob(os.path.join(class_dir, "*.*"))
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                objs = []
                for f in image_files:
                    img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        objs.append(img)
                objects_by_class[cls] = objs

            # Calcular conteos originales
            original_counts = {}
            for ann in coco_data["annotations"]:
                cat_name = categories.get(ann["category_id"], "")
                if cat_name in selected_classes:
                    original_counts[cat_name] = original_counts.get(cat_name, 0) + 1
            if not original_counts:
                logging.warning("No se encontraron instancias originales para las clases seleccionadas.")
                return synthetic_counts_by_class, synthetic_count
            target_count = max(original_counts.values())
            required_instances = {cls: max(target_count - original_counts.get(cls, 0), 0)
                                for cls in selected_classes}
            if desired_synthetic_per_class is not None:
                required_instances = {cls: int(desired_synthetic_per_class.get(cls, 0)) for cls in selected_classes}
            initial_required = sum(required_instances.values())
            logging.info(f"Target count: {target_count}, Instancias requeridas: {required_instances}")
            
            bg_files = [f for f in glob.glob(os.path.join(backgrounds_dataset_path, "*.*"))
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not bg_files:
                logging.error("No se encontraron fondos en el directorio especificado.")
                return synthetic_counts_by_class, synthetic_count
            annotated_dir = os.path.join(self.output_dir, "annotated")
            ensure_dir(annotated_dir)
            
            while any(count > 0 for count in required_instances.values()):
                total_required = sum(required_instances.values())
                num_objetos = random.randint(1, min(max_objects_per_image, total_required))
                logging.debug(f"Generando imagen sintética con {num_objetos} objetos.")
                available_classes = [cls for cls, req in required_instances.items() if req > 0]
                if not available_classes:
                    break
                if num_objetos > len(available_classes):
                    selected_cls_list = random.choices(available_classes, k=num_objetos)
                else:
                    selected_cls_list = random.sample(available_classes, k=num_objetos)
                bg_file = random.choice(bg_files)
                bg_image = cv2.imread(bg_file)
                if bg_image is None:
                    logging.error(f"Error al cargar fondo {bg_file}")
                    continue
                annotation_shapes = []
                bg_bin = np.zeros_like(bg_image, dtype=np.uint8)
                # Usar máscara acumulada para controlar overlapping
                existing_mask = np.zeros(bg_image.shape[:2], dtype=np.uint8)
                for cls in selected_cls_list:
                    obj_list = objects_by_class.get(cls, [])
                    if not obj_list:
                        logging.warning(f"No se encontraron objetos para {cls} en la carpeta.")
                        continue
                    obj_image = random.choice(obj_list)
                    result = self.paste_object(bg_bin, bg_image, obj_image, idx_obj=1,
                                                existing_mask=existing_mask, max_overlap_threshold=0.2)
                    (bg_bin, bg_image, angle, pos, scale_factor, patched,
                    eff_hp, eff_wp, candidate_box, updated_mask) = result
                    if patched and candidate_box is not None:
                        existing_mask = updated_mask
                        x, y, x2, y2 = candidate_box
                        shape = {
                            "label": cls,
                            "points": [[x, y], [x2, y], [x2, y2], [x, y2]],
                            "group_id": None,
                            "shape_type": "rectangle",
                            "flags": {}
                        }
                        annotation_shapes.append(shape)
                        required_instances[cls] = max(required_instances[cls] - 1, 0)
                        synthetic_counts_by_class[cls] += 1
                    else:
                        logging.warning(f"No se pudo pegar objeto para {cls} en este intento.")
                if annotation_shapes:
                    image_name = f"synthetic_{synthetic_count}.jpg"
                    output_image_path = os.path.join(self.images_dir, image_name)
                    cv2.imwrite(output_image_path, bg_image)
                    logging.info(f"Imagen sintética guardada: {output_image_path}")
                    annotation = {
                        "version": "4.5.7",
                        "flags": {},
                        "shapes": annotation_shapes,
                        "imagePath": image_name,
                        "imageData": None,
                        "imageHeight": bg_image.shape[0],
                        "imageWidth": bg_image.shape[1]
                    }
                    try:
                        with open(output_image_path, "rb") as img_file:
                            encoded = base64.b64encode(img_file.read()).decode("utf-8")
                        annotation["imageData"] = encoded
                    except Exception as e:
                        logging.error(f"Error al codificar {output_image_path}: {e}")
                    json_name = image_name.replace(".jpg", ".json")
                    dest_json_path = os.path.join(self.labels_dir, json_name)
                    try:
                        with open(dest_json_path, "w") as f:
                            json.dump(annotation, f, indent=2, separators=(", ", ": "))
                        logging.info(f"Anotación guardada: {dest_json_path}")
                    except Exception as e:
                        logging.error(f"Error al guardar anotación en {dest_json_path}: {e}")
                    synthetic_count += 1
                    annotated_image = self.draw_annotations_on_image(bg_image, annotation_shapes)
                    annotated_image_path = os.path.join(annotated_dir, image_name)
                    cv2.imwrite(annotated_image_path, annotated_image)
                    logging.info(f"Imagen anotada guardada: {annotated_image_path}")
                else:
                    logging.debug("No se pegaron objetos en esta iteración.")
                if progress_bar is not None and initial_required > 0:
                    current_required = sum(required_instances.values())
                    progress = (initial_required - current_required) / initial_required
                    progress_bar.progress(min(progress, 1.0))
            logging.info("Proceso de augmentación en modo 'folder' completado.")
            return synthetic_counts_by_class, synthetic_count

        else:
            logging.info("Modo tradicional: procesando imágenes originales.")
            return synthetic_counts_by_class, synthetic_count

        # Al finalizar, generar un JSON global en formato COCO con todas las anotaciones sintéticas.
        all_images = []
        all_annotations = []
        for i, file in enumerate(sorted(os.listdir(self.labels_dir))):
            if file.endswith(".json"):
                json_path = os.path.join(self.labels_dir, file)
                with open(json_path, "r") as f:
                    ann_data = json.load(f)
                img_info = {
                    "id": i + 1,
                    "file_name": ann_data["imagePath"],
                    "width": ann_data["imageWidth"],
                    "height": ann_data["imageHeight"]
                }
                all_images.append(img_info)
                for shape in ann_data["shapes"]:
                    x1, y1 = shape["points"][0]
                    x2, y2 = shape["points"][2]
                    bbox = [x1, y1, x2 - x1, y2 - y1]
                    ann_info = {
                        "id": len(all_annotations) + 1,
                        "image_id": i + 1,
                        "category_id": self.category_map.get(shape["label"], 0),
                        "segmentation": [],
                        "area": bbox[2] * bbox[3],
                        "bbox": bbox,
                        "iscrowd": 0
                    }
                    all_annotations.append(ann_info)
        categories_list = []
        for name, cat_id in self.category_map.items():
            categories_list.append({
                "id": cat_id,
                "name": name,
                "supercategory": "none"
            })
        coco_synthetic = {
            "info": {
                "description": "Synthetic Dataset",
                "version": "1.0",
                "year": 2025,
                "contributor": "",
                "date_created": ""
            },
            "licenses": [],
            "images": all_images,
            "annotations": all_annotations,
            "categories": categories_list
        }
        output_json_path = os.path.join(self.output_dir, "synthetic_dataset.json")
        with open(output_json_path, "w") as f:
            json.dump(coco_synthetic, f, indent=2)
        logging.info(f"Synthetic COCO JSON guardado en {output_json_path}")
        return synthetic_counts_by_class, synthetic_count