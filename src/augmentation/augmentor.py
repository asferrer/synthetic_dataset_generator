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

COLOR_PALETTE = [
    (0, 255, 0),    # Verde brillante
    (0, 0, 255),    # Rojo
    (255, 0, 0),    # Azul
    (0, 255, 255),  # Cian/Amarillo verdoso
    (255, 0, 255),  # Magenta
    (255, 255, 0),  # Amarillo/Azul verdoso
    (0, 128, 255),  # Naranja
    (128, 0, 128),  # Púrpura
    (60, 180, 75),  # Verde lima
    (245, 130, 48), # Naranja oscuro
    (0, 130, 200),  # Azul acero
    (145, 30, 180), # Orquídea
    (70, 240, 240), # Turquesa
    (240, 50, 230), # Rosa neón
    (210, 245, 60), # Verde amarillento
    (250, 190, 190),# Rosa pálido
    (0, 128, 128),  # Teal
    (230, 190, 255),# Lavanda
    (170, 110, 40), # Marrón
    (255, 250, 200),# Crema
    (128, 0, 0),    # Marrón oscuro/Granate
    (170, 255, 195),# Menta
    (128, 128, 0),  # Oliva
    (255, 215, 180),# Melocotón
    (0, 0, 128),    # Azul marino
    (128, 128, 128),# Gris
    (255, 255, 255),# Blanco (menos útil sobre fondos claros)
    (0, 0, 0),      # Negro (útil para texto sobre colores claros)
]

def compute_iou(boxA, boxB):
    # Calcula la intersección sobre la unión (IoU) de dos bounding boxes
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
                 overlap_threshold=0.1,
                 seed=1,
                 save_intermediate_steps=False):
        self.rot = rot
        self.scale = scale
        self.trans = trans
        self.try_count = try_count
        self.overlap_threshold = overlap_threshold
        self.output_dir = output_dir
        random.seed(seed)
        self.save_intermediate_steps = save_intermediate_steps

        self.images_dir = os.path.join(self.output_dir, "images")
        self.labels_dir = os.path.join(self.output_dir, "labels")
        self.annotated_dir = os.path.join(self.output_dir, "annotated")


        ensure_dir(self.images_dir)
        ensure_dir(self.labels_dir)
        ensure_dir(self.annotated_dir)

        # Crear directorios para pasos intermedios si está activado
        if self.save_intermediate_steps:
            self.intermediate_dir = os.path.join(self.output_dir, "intermediate_steps")
            self.backgrounds_out_dir = os.path.join(self.intermediate_dir, "01_backgrounds")
            self.objects_out_dir = os.path.join(self.intermediate_dir, "02_selected_objects")
            self.pasted_raw_dir = os.path.join(self.intermediate_dir, "03_pasted_raw")
            self.pasted_blended_dir = os.path.join(self.intermediate_dir, "04_pasted_blended")

            ensure_dir(self.intermediate_dir)
            ensure_dir(self.backgrounds_out_dir)
            ensure_dir(self.objects_out_dir)
            ensure_dir(self.pasted_raw_dir)
            ensure_dir(self.pasted_blended_dir)

        logging.debug(f"Directorios creados: {self.images_dir}, {self.labels_dir}, {self.annotated_dir}")
        if self.save_intermediate_steps:
            logging.debug(f"Directorios intermedios creados en: {self.intermediate_dir}")

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

        # --- Calcular dimensiones efectivas y validar posición ---
        # Asegurarse de que la posición inicial sea válida
        if x >= bg_w or y >= bg_h or x + wp <= 0 or y + hp <= 0:
             logging.warning(f"Patch con pos {pos} y dims ({hp},{wp}) está completamente fuera del fondo ({bg_h},{bg_w}).")
             return bg.copy(), bg.copy(), 0, 0 # Devolver originales, dimensiones 0

        # Calcular solapamiento real y ajustar coordenadas/dimensiones
        x_start_bg = max(x, 0)
        y_start_bg = max(y, 0)
        x_end_bg = min(x + wp, bg_w)
        y_end_bg = min(y + hp, bg_h)

        effective_wp = x_end_bg - x_start_bg
        effective_hp = y_end_bg - y_start_bg

        # Calcular coordenadas de inicio en el *patch* original
        x_start_patch = max(0, -x)
        y_start_patch = max(0, -y)

        # --- Validar dimensiones efectivas ---
        if effective_hp <= 0 or effective_wp <= 0:
            logging.warning(f"Calculadas dimensiones efectivas inválidas ({effective_hp}, {effective_wp}) para pos {pos}. No se puede pegar.")
            return bg.copy(), bg.copy(), 0, 0

        # --- Recortar el patch a la parte que realmente se solapa ---
        try:
             patch_clipped = patch[y_start_patch:y_start_patch + effective_hp,
                                   x_start_patch:x_start_patch + effective_wp, :]
        except IndexError as e:
             logging.error(f"Error al recortar patch: {e}. Patch shape: {patch.shape}, "
                           f"y_start: {y_start_patch}, y_end: {y_start_patch + effective_hp}, "
                           f"x_start: {x_start_patch}, x_end: {x_start_patch + effective_wp}")
             return bg.copy(), bg.copy(), 0, 0


        # --- Asegurar que el patch recortado tenga canal alfa ---
        if patch_clipped.shape[2] < 4:
            # Manejar diferentes números de canales de entrada
            if patch_clipped.shape[2] == 1: # Grayscale
                 patch_clipped_bgr = cv2.cvtColor(patch_clipped, cv2.COLOR_GRAY2BGR)
            elif patch_clipped.shape[2] == 3: # BGR
                 patch_clipped_bgr = patch_clipped
            else: # Caso inesperado
                 logging.error(f"Número de canales inesperado en patch_clipped: {patch_clipped.shape[2]}")
                 return bg.copy(), bg.copy(), 0, 0

            mask = 255 * np.ones((effective_hp, effective_wp), dtype=np.uint8)
            patch_rgba = cv2.cvtColor(patch_clipped_bgr, cv2.COLOR_BGR2BGRA)
            patch_rgba[:, :, 3] = mask
            patch_clipped = patch_rgba # Ahora es BGRA (effective_hp, effective_wp, 4)
        elif patch_clipped.shape[2] > 4: # Si tiene más de 4 canales, tomar los primeros 4
             patch_clipped = patch_clipped[:, :, :4]


        # --- Separar canales del patch recortado ---
        try:
            b, g, r, a = cv2.split(patch_clipped)
        except cv2.error as e:
             logging.error(f"Error en cv2.split: {e}. Shape de patch_clipped: {patch_clipped.shape}")
             return bg.copy(), bg.copy(), 0, 0

        patch_bgr_original = cv2.merge((b, g, r)) # Shape: (effective_hp, effective_wp, 3)
        alpha_original_norm = a.astype(np.float32) / 255.0 # Shape: (effective_hp, effective_wp)

        # --- Definir la ROI en el background usando coordenadas ajustadas ---
        roi_slice_y = slice(y_start_bg, y_end_bg) # Equivalente a y_start_bg : y_start_bg + effective_hp
        roi_slice_x = slice(x_start_bg, x_end_bg) # Equivalente a x_start_bg : x_start_bg + effective_wp

        # --- 1. Pegado Raw ---
        bg_raw_paste = bg.copy()
        try:
             roi_raw = bg_raw_paste[roi_slice_y, roi_slice_x]
        except IndexError as e:
             logging.error(f"Error al extraer ROI raw: {e}. bg shape: {bg_raw_paste.shape}, slice_y: {roi_slice_y}, slice_x: {roi_slice_x}")
             return bg.copy(), bg.copy(), 0, 0

        # Validar forma de ROI Raw
        if roi_raw.shape[:2] != (effective_hp, effective_wp):
             logging.error(f"Discrepancia de formas - ROI Raw: {roi_raw.shape[:2]} vs Patch Efectivo: ({effective_hp}, {effective_wp})")
             return bg.copy(), bg.copy(), 0, 0

        alpha_3ch_orig = cv2.merge([alpha_original_norm] * 3) # Shape: (effective_hp, effective_wp, 3)

        # Mezcla Raw
        try:
            raw_blended_area = patch_bgr_original.astype(np.float32) * alpha_3ch_orig + \
                               roi_raw.astype(np.float32) * (1 - alpha_3ch_orig)
            bg_raw_paste[roi_slice_y, roi_slice_x] = np.clip(raw_blended_area, 0, 255).astype(np.uint8)
            logging.debug(f"Raw patch pasted at bg coords ({y_start_bg}:{y_end_bg}, {x_start_bg}:{x_end_bg})")
        except ValueError as e:
            logging.error(f"Error de broadcasting en pegado raw: {e}")
            logging.error(f"Shapes - patch_bgr: {patch_bgr_original.shape}, alpha3ch: {alpha_3ch_orig.shape}, roi_raw: {roi_raw.shape}")
            return bg.copy(), bg.copy(), 0, 0 # Error


        # --- 2. Adaptación de Tono ---
        roi_for_tone = bg[roi_slice_y, roi_slice_x] # Usar ROI del fondo original
        # Validar forma de ROI Tone (debería ser igual que roi_raw)
        if roi_for_tone.shape[:2] != (effective_hp, effective_wp):
             logging.error(f"Discrepancia de formas - ROI Tone: {roi_for_tone.shape[:2]} vs Patch Efectivo: ({effective_hp}, {effective_wp})")
             return bg.copy(), bg_raw_paste, 0, 0 # Devolver raw ya calculado

        # Calcular media del fondo en ROI usando máscara 'a'
        bg_mean = cv2.mean(roi_for_tone, mask=a)[:3]

        # Calcular media del objeto (ya se usa patch_clipped con forma correcta)
        obj_mean = []
        for channel in (b, g, r):
            valid_pixels = channel[alpha_original_norm > 0]
            channel_mean = np.mean(valid_pixels) if valid_pixels.size > 0 else 0
            obj_mean.append(channel_mean)
        obj_mean = np.array(obj_mean)
        bg_mean = np.array(bg_mean)
        diff = bg_mean - obj_mean

        # Aplicar diferencia de tono
        b_adapted = np.clip(b.astype(np.float32) + diff[0], 0, 255).astype(np.uint8)
        g_adapted = np.clip(g.astype(np.float32) + diff[1], 0, 255).astype(np.uint8)
        r_adapted = np.clip(r.astype(np.float32) + diff[2], 0, 255).astype(np.uint8)
        patch_bgr_adapted = cv2.merge((b_adapted, g_adapted, r_adapted)) # Shape: (effective_hp, effective_wp, 3)
        logging.debug(f"Tone adapted with diff: {diff}")


        # --- 3. Suavizado de Bordes (Blurring Alpha) ---
        blur_kernel_size = min(7, max(3, int(min(effective_hp, effective_wp) * 0.1) // 2 * 2 + 1))
        blurred_alpha = cv2.GaussianBlur(a, (blur_kernel_size, blur_kernel_size), 0) # Shape: (effective_hp, effective_wp)
        blurred_alpha_norm = blurred_alpha.astype(np.float32) / 255.0
        alpha_3ch_blurred = cv2.merge([blurred_alpha_norm] * 3) # Shape: (effective_hp, effective_wp, 3)
        logging.debug(f"Alpha blurred with kernel size: {blur_kernel_size}")


        # --- 4. Pegado Final Blended ---
        bg_final_paste = bg.copy()
        # roi_final es la misma región que roi_raw/roi_for_tone, pero en bg_final_paste
        roi_final = bg_final_paste[roi_slice_y, roi_slice_x]

        # Validar forma de ROI Final
        if roi_final.shape[:2] != (effective_hp, effective_wp):
             logging.error(f"Discrepancia de formas - ROI Final: {roi_final.shape[:2]} vs Patch Efectivo: ({effective_hp}, {effective_wp})")
             return bg_final_paste, bg_raw_paste, 0, 0 # Devolver lo que se tenga

        # Mezcla Final Blended
        try:
            final_blended_area = patch_bgr_adapted.astype(np.float32) * alpha_3ch_blurred + \
                                 roi_final.astype(np.float32) * (1 - alpha_3ch_blurred)
            bg_final_paste[roi_slice_y, roi_slice_x] = np.clip(final_blended_area, 0, 255).astype(np.uint8)
            logging.debug(f"Final blended patch pasted at bg coords ({y_start_bg}:{y_end_bg}, {x_start_bg}:{x_end_bg})")
        except ValueError as e:
            logging.error(f"Error de broadcasting en pegado final: {e}")
            logging.error(f"Shapes - patch_adapt: {patch_bgr_adapted.shape}, alpha3ch_blur: {alpha_3ch_blurred.shape}, roi_final: {roi_final.shape}")
            return bg.copy(), bg_raw_paste, 0, 0 # Error, devolver raw calculado


        # Devolver blended, raw, y las dimensiones efectivas *reales* usadas
        return bg_final_paste, bg_raw_paste, effective_hp, effective_wp

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

    def paste_object(self, bg_bin, bg_image, bg_image_raw, # Añadir bg_image_raw como entrada/salida
                     patch, idx_obj, existing_mask=None,
                     max_overlap_threshold=None, existing_bboxes=None):
        """
        Intenta pegar un objeto sobre el fondo controlando el solapamiento [...]
        Ahora también maneja y devuelve la imagen con pegado "raw".
        [...]
        Retorna:
        - bg_bin: máscara actualizada.
        - bg_image_blended: imagen de fondo actualizada con blending.
        - bg_image_raw: imagen de fondo actualizada con pegado raw. # Nuevo retorno
        - final_angle, final_pos, final_scale: parámetros de transformación aplicados.
        - patched: booleano que indica si el objeto fue pegado.
        - effective_hp, effective_wp: dimensiones efectivas del área pegada.
        - candidate_box: bounding box del candidato.
        - updated_mask: máscara acumulada actualizada.
        - updated_bboxes: lista actualizada de bounding boxes.
        """
        if max_overlap_threshold is None:
            max_overlap_threshold = self.overlap_threshold
        if existing_bboxes is None:
            existing_bboxes = []

        # Hacer copias para no modificar los originales si falla el pegado
        bg_image_blended_candidate = bg_image.copy()
        bg_image_raw_candidate = bg_image_raw.copy()
        bg_bin_candidate_state = bg_bin.copy() # Para restaurar si falla un intento
        updated_mask_candidate = existing_mask.copy() if existing_mask is not None else np.zeros(bg_image.shape[:2], dtype=np.uint8)
        current_bboxes_candidate = list(existing_bboxes) # Copia de bboxes

        patched = False
        final_angle, final_pos, final_scale = 0, (0, 0), 1.0
        effective_hp, effective_wp = 0, 0
        candidate_box = None
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) # No parece usarse

        for i in range(self.try_count):
            transformed_patch, angle, scale_factor, pos = self.apply_transformations(patch, bg_image.shape)

            # Verificar si el patch transformado tiene dimensiones válidas
            if transformed_patch.shape[0] == 0 or transformed_patch.shape[1] == 0:
                 logging.warning(f"Intento {i+1}: Patch transformado inválido (dimensión 0). Reintentando.")
                 continue

            # --- Calcular máscara y BBox del candidato ---
            if transformed_patch.shape[2] < 4:
                candidate_mask = np.ones(transformed_patch.shape[:2], dtype=np.uint8) * 255
            else:
                candidate_mask = transformed_patch[:, :, 3]

            # Asegurar que el mask tenga las dimensiones correctas antes del threshold
            if candidate_mask.shape[0] == 0 or candidate_mask.shape[1] == 0:
                 logging.warning(f"Intento {i+1}: Máscara del candidato inválida (dimensión 0). Reintentando.")
                 continue

            _, candidate_mask_bin = cv2.threshold(candidate_mask, 127, 255, cv2.THRESH_BINARY)

            bg_h, bg_w = bg_image.shape[:2]
            # Calcular dimensiones efectivas al pegar (considerando límites del fondo)
            effective_h_paste = min(transformed_patch.shape[0], bg_h - pos[1])
            effective_w_paste = min(transformed_patch.shape[1], bg_w - pos[0])

            # Validar dimensiones efectivas
            if effective_h_paste <= 0 or effective_w_paste <= 0:
                logging.debug(f"Intento {i+1}: Posición {pos} inválida para patch. Reintentando.")
                continue

            candidate_box = (pos[0], pos[1], pos[0] + effective_w_paste, pos[1] + effective_h_paste)

            # --- Evaluar solapamiento con BBoxes existentes ---
            overlap_flag = False
            for bb in current_bboxes_candidate: # Usar bboxes del estado candidato actual
                iou = compute_iou(candidate_box, bb)
                if iou > max_overlap_threshold:
                    overlap_flag = True
                    logging.debug(f"Intento {i+1}: IoU {iou:.2f} con BB existente {bb} supera umbral {max_overlap_threshold}. Reintentando.")
                    break

            if overlap_flag:
                continue  # Reintentar sin pegar

            # --- Si no hay solapamiento, realizar el pegado (raw y blended) ---
            # Usar las copias candidatas para el pegado
            pasted_blended, pasted_raw, effective_hp, effective_wp = self._paste_png(
                bg_image_blended_candidate, transformed_patch, pos
            )
            # La versión raw necesita su propio pegado independiente partiendo del estado raw anterior
            _, pasted_raw_only, _, _ = self._paste_png(
                 bg_image_raw_candidate, transformed_patch, pos
            )


            # Validar dimensiones efectivas devueltas por _paste_png
            if effective_hp <= 0 or effective_wp <= 0:
                 logging.warning(f"Intento {i+1}: _paste_png devolvió dimensiones inválidas ({effective_hp}, {effective_wp}). Reintentando.")
                 continue

            # Actualizar BBox con dimensiones efectivas reales del pegado
            candidate_box = (pos[0], pos[1], pos[0] + effective_wp, pos[1] + effective_hp)

            # Actualizar máscara binaria de ocupación (bg_bin)
            bg_bin_candidate, _ = self._paste_bin_png(bg_bin_candidate_state, transformed_patch, idx_obj, pos) # Usar estado binario anterior

            # Actualizar máscara de segmentación acumulada
            candidate_mask_resized = candidate_mask_bin[:effective_hp, :effective_wp]
            candidate_mask_pasted = np.zeros(bg_image.shape[:2], dtype=np.uint8)
            candidate_mask_pasted[pos[1]:pos[1]+effective_hp, pos[0]:pos[0]+effective_wp] = candidate_mask_resized
            updated_mask_candidate = cv2.bitwise_or(updated_mask_candidate, candidate_mask_pasted)


            # Actualizar estados con los resultados del pegado exitoso
            bg_image_blended_candidate = pasted_blended
            bg_image_raw_candidate = pasted_raw_only # Actualizar con el resultado raw
            bg_bin_candidate_state = bg_bin_candidate # Actualizar estado binario
            current_bboxes_candidate.append(candidate_box) # Añadir nueva BBox

            final_angle, final_pos, final_scale = angle, pos, scale_factor
            patched = True
            logging.debug(f"Intento {i+1}: Objeto pegado exitosamente. Pos: {pos}, Scale: {scale_factor:.2f}, Angle: {angle:.1f}")
            break # Salir del bucle de intentos si se pegó bien

        if not patched:
            logging.warning(f"No se pudo pegar el objeto (clase idx {idx_obj}) tras {self.try_count} intentos.")
            # Devolver los estados originales si no se pudo pegar nada
            return bg_bin, bg_image, bg_image_raw, 0, (0, 0), 1.0, False, 0, 0, None, existing_mask, existing_bboxes

        # Devolver los estados actualizados si el pegado fue exitoso
        return (bg_bin_candidate_state, bg_image_blended_candidate, bg_image_raw_candidate,
                final_angle, final_pos, final_scale, patched,
                effective_hp, effective_wp, candidate_box, updated_mask_candidate, current_bboxes_candidate)

    def draw_annotations_on_image(self, image, shapes):
        annotated = image.copy()

        if not shapes: # Si no hay shapes, devolver la imagen original
            return annotated
        
        try:
            unique_labels = sorted(list(set(shape['label'] for shape in shapes if 'label' in shape)))
        except KeyError:
             logging.error("Error al extraer etiquetas de 'shapes'. Asegúrate de que cada shape tenga una 'label'.")
             unique_labels = [] # Continuar sin colores específicos o devolver error

        if not unique_labels:
             logging.warning("No se encontraron etiquetas en las shapes proporcionadas.")

        color_map = {}
        palette_size = len(COLOR_PALETTE)

        for i, label in enumerate(unique_labels):
            # Asignar colores de la paleta cíclicamente
            color_map[label] = COLOR_PALETTE[i % palette_size]
            logging.debug(f"Mapeando etiqueta '{label}' al color {color_map[label]}")

        for shape in shapes:
            label = shape.get("label", "unknown")
            points = shape.get("points", [])

            color = color_map.get(label, (255, 255, 255)) # Blanco por defecto

            if shape.get("shape_type") == "rectangle" and len(points) >= 2:
                try:
                    # Extraer coordenadas y convertir a enteros
                    # Asegurarse de que los puntos sean TL y BR correctamente
                    all_x = [p[0] for p in points]
                    all_y = [p[1] for p in points]
                    x1, y1 = min(all_x), min(all_y)
                    x2, y2 = max(all_x), max(all_y)
                    top_left = (int(x1), int(y1))
                    bottom_right = (int(x2), int(y2))

                    # Dibujar el rectángulo
                    cv2.rectangle(annotated, top_left, bottom_right, color, thickness=2)

                    # Preparar y dibujar el texto de la etiqueta
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    text_thickness = 1 # Usar 1 para mejor legibilidad en texto pequeño
                    text = label
                    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)

                    # Posición del texto (arriba de la esquina superior izquierda)
                    text_x = top_left[0]
                    text_y = top_left[1] - 10

                    # Ajustar si el texto se sale por arriba
                    if text_y < text_height:
                        text_y = top_left[1] + text_height + baseline + 5 # Ponerlo debajo

                    # Poner un fondo pequeño para el texto para mejorar legibilidad (opcional)
                    # text_bg_y2 = text_y + baseline # Altura real del texto
                    # text_bg_x2 = text_x + text_width
                    # cv2.rectangle(annotated, (text_x, text_y - text_height - baseline), (text_bg_x2, text_bg_y2), (0,0,0), cv2.FILLED) # Fondo negro

                    cv2.putText(annotated, text, (text_x, text_y),
                                font, font_scale, color, text_thickness, lineType=cv2.LINE_AA)

                except (ValueError, IndexError, TypeError) as e:
                    logging.warning(f"No se pudo dibujar el rectángulo para la etiqueta '{label}' con puntos {points}: {e}")
            else:
                logging.warning(f"Saltando shape para etiqueta '{label}': no es rectángulo o tiene puntos insuficientes ({len(points)}). Shape: {shape}")

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
        Genera imágenes sintéticas asegurando que se crean las muestras necesarias para cada clase,
        sin saturar las imágenes (máximo max_objects_per_image objetos por imagen).
        El proceso continúa hasta que todas las clases tengan el número requerido de muestras.
        
        Parámetros:
        - coco_data: Diccionario del dataset COCO de entrada.
        - images_path: Ruta de las imágenes originales.
        - selected_classes: Lista de nombres de clases a aumentar.
        - objects_source: "dataset" para usar objetos segmentados del dataset o "folder" para usar objetos externos.
        - objects_dataset_path: Ruta a la carpeta con objetos (si se usa "folder").
        - backgrounds_dataset_path: Ruta a la carpeta que contiene fondos.
        - max_objects_per_image: Número máximo de objetos que se pegarán en cada imagen sintética.
        - desired_synthetic_per_class: Diccionario con el número deseado de muestras sintéticas por clase.
        - progress_bar: (Opcional) Objeto de progreso (por ejemplo, de Streamlit).
        
        Retorna:
        - synthetic_counts_by_class: Diccionario con la cantidad de instancias generadas por cada clase.
        - synthetic_total: Total de imágenes sintéticas generadas.
        
        Al finalizar se genera un JSON global en formato COCO, en el cual:
          - Los IDs de categoría comienzan en 0.
          - Todas las anotaciones usan el nuevo mapeo de categorías.
          
        Además, se almacenan en la carpeta "annotated" las imágenes sintéticas con las anotaciones dibujadas.
        """
        synthetic_counts_by_class = {cls: 0 for cls in selected_classes}
        synthetic_total = 0

        # Calcular instancias originales por clase
        categories = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
        original_counts = {}
        for ann in coco_data["annotations"]:
            cat_name = categories.get(ann["category_id"], "")
            if cat_name in selected_classes:
                original_counts[cat_name] = original_counts.get(cat_name, 0) + 1

        if desired_synthetic_per_class is not None:
            required_instances = {cls: int(desired_synthetic_per_class.get(cls, 0)) for cls in selected_classes}
        else:
            target_count = max(original_counts.values()) if original_counts else 0
            required_instances = {cls: max(target_count - original_counts.get(cls, 0), 0) for cls in selected_classes}
        logging.info(f"Instancias sintéticas requeridas por clase: {required_instances}")

        ensure_dir(self.images_dir)
        ensure_dir(self.labels_dir)
        annotated_dir = os.path.join(self.output_dir, "annotated")
        ensure_dir(annotated_dir)

        # Nuevo mapeo de categorías: IDs consecutivos iniciando en 0
        sorted_classes = sorted(selected_classes)
        self.category_map = {cls: idx for idx, cls in enumerate(sorted_classes)}
        logging.info(f"Nuevo mapeo de categorías (IDs empiezan en 0): {self.category_map}")

        if not backgrounds_dataset_path:
            logging.error("Se requiere un directorio de fondos.")
            return synthetic_counts_by_class, synthetic_total
        bg_files = [f for f in glob.glob(os.path.join(backgrounds_dataset_path, "*.*"))
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not bg_files:
            logging.error("No se encontraron fondos en el directorio especificado.")
            return synthetic_counts_by_class, synthetic_total

        # Cargar objetos según el modo
        objects_by_class = {}
        if objects_source == "dataset":
            logging.info("Modo 'dataset' activado: utilizando objetos segmentados del dataset de entrada.")
            segmentation_mode = any("segmentation" in ann and ann["segmentation"] for ann in coco_data["annotations"])
            if not segmentation_mode:
                logging.error("No se detectaron anotaciones de segmentación en el dataset.")
                return synthetic_counts_by_class, synthetic_total
            objects_by_class = self.extract_segmented_objects(coco_data, images_path, selected_classes)
        elif objects_source == "folder" and objects_dataset_path:
            logging.info("Modo 'folder' activado: utilizando objetos de carpeta externa.")
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
        else:
            logging.error("No se ha especificado un modo válido para cargar objetos.")
            return synthetic_counts_by_class, synthetic_total

        def update_progress():
            if progress_bar is not None and desired_synthetic_per_class:
                total_remaining = sum(required_instances.values())
                total_initial = sum(int(desired_synthetic_per_class.get(cls, 0)) for cls in selected_classes)
                if total_initial > 0:
                    progress = 1 - (total_remaining / total_initial)
                    progress_bar.progress(min(progress, 1.0))

        # Bucle global: generar imágenes hasta que TODAS las clases tengan sus muestras requeridas
        while any(required_instances[cls] > 0 for cls in selected_classes):
            bg_file = random.choice(bg_files)
            bg_image = cv2.imread(bg_file)

            if bg_image is None:
                logging.error(f"Error al cargar fondo: {bg_file}")
                if len(bg_files) > 1: bg_files.remove(bg_file) # Evitar reintentar con el mismo fondo malo
                else: break # Salir si no quedan fondos válidos
                continue

            # Nombre base para esta imagen sintética
            image_base_name = f"synthetic_{synthetic_total}"
            image_name = f"{image_base_name}.jpg"

            # Guardar background original si está activado
            if self.save_intermediate_steps:
                bg_out_path = os.path.join(self.backgrounds_out_dir, f"{image_base_name}_background.jpg")
                cv2.imwrite(bg_out_path, bg_image)
                logging.debug(f"Intermediate step saved: {bg_out_path}")
            
            bg_image_raw = bg_image.copy() 
            bg_bin = np.zeros_like(bg_image, dtype=np.uint8)
            existing_mask = np.zeros(bg_image.shape[:2], dtype=np.uint8)
            current_bboxes = []  # Lista de BB ya pegadas en esta imagen
            annotation_shapes = []

            total_required = sum(required_instances[cls] for cls in selected_classes)
            
            if total_required == 0:
                logging.info("Ya no se requieren más objetos sintéticos. Finalizando generación.")
                break

            num_objetos = random.randint(1, min(max_objects_per_image, total_required))

            available_classes = [cls for cls in selected_classes if required_instances[cls] > 0]
            if num_objetos > len(available_classes):
                selected_cls_list = random.choices(available_classes, k=num_objetos)
            else:
                selected_cls_list = random.sample(available_classes, k=num_objetos)

            for obj_idx, cls in enumerate(selected_cls_list):
                objs = objects_by_class.get(cls, [])
                if not objs:
                    logging.warning(f"No se encontraron objetos para la clase {cls}.")
                    continue
                obj_image = random.choice(objs).copy()

                result = self.paste_object(bg_bin, bg_image, bg_image_raw, obj_image, idx_obj=1,
                                           existing_mask=existing_mask,
                                           max_overlap_threshold=self.overlap_threshold,
                                           existing_bboxes=current_bboxes)
                
                (bg_bin, bg_image, bg_image_raw, angle, pos, scale_factor, patched,
                 eff_hp, eff_wp, candidate_box, updated_mask, current_bboxes) = result

                if patched and candidate_box is not None:
                    if self.save_intermediate_steps:
                        obj_filename = f"{image_base_name}_object_{obj_idx}_{cls}.png"
                        obj_out_path = os.path.join(self.objects_out_dir, obj_filename)
                        try:
                            cv2.imwrite(obj_out_path, obj_image)
                            logging.debug(f"Intermediate step saved: Individual object {obj_out_path}")
                        except Exception as e:
                            logging.error(f"Error al guardar objeto individual {obj_out_path}: {e}")
                                
                    x, y, x2, y2 = candidate_box
                    shape = {
                        "label": cls,
                        "points": [[x, y], [x2, y], [x2, y2], [x, y2]],
                        "group_id": None,
                        "shape_type": "rectangle",
                        "flags": {}
                    }
                    annotation_shapes.append(shape)
                    existing_mask = updated_mask
                    required_instances[cls] -= 1
                    synthetic_counts_by_class[cls] += 1
                    logging.debug(f"Objeto de clase '{cls}' pegado. Restantes para la clase: {required_instances[cls]}")
                else:
                    logging.debug(f"No se pudo pegar el objeto para la clase {cls} en este intento.")

            if annotation_shapes:
                output_image_path = os.path.join(self.images_dir, image_name)
                cv2.imwrite(output_image_path, bg_image)
                logging.info(f"Imagen sintética guardada: {output_image_path}")

                if self.save_intermediate_steps:
                    # Guardar imagen con pegado raw acumulado
                    raw_filename = f"{image_base_name}_pasted_raw.jpg"
                    raw_out_path = os.path.join(self.pasted_raw_dir, raw_filename)
                    try:
                        # Guardar la imagen raw acumulada después de todos los pegados
                        cv2.imwrite(raw_out_path, bg_image_raw)
                        logging.debug(f"Intermediate step saved: {raw_out_path}")
                    except Exception as e:
                        logging.error(f"Error al guardar imagen raw intermedia {raw_out_path}: {e}")

                    # Guardar imagen con pegado blended acumulado (opcional, es igual a la de images_dir)
                    blended_filename = f"{image_base_name}_pasted_blended.jpg"
                    blended_out_path = os.path.join(self.pasted_blended_dir, blended_filename)
                    try:
                        # Guardar la imagen blended acumulada (igual que la guardada en images_dir)
                        cv2.imwrite(blended_out_path, bg_image)
                        logging.debug(f"Intermediate step saved: {blended_out_path}")
                    except Exception as e:
                        logging.error(f"Error al guardar imagen blended intermedia {blended_out_path}: {e}")

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

                # Dibujar las anotaciones sobre la imagen y guardarla en "annotated"
                annotated_image = self.draw_annotations_on_image(bg_image, annotation_shapes)
                annotated_image_path = os.path.join(annotated_dir, image_name)
                cv2.imwrite(annotated_image_path, annotated_image)
                logging.info(f"Imagen anotada guardada: {annotated_image_path}")

                synthetic_total += 1
                update_progress()

        # Generación del JSON global COCO
        all_images = []
        all_annotations = []
        annotation_files = sorted([f for f in os.listdir(self.labels_dir) if f.endswith(".json")])
        for i, file in enumerate(annotation_files):
            json_path = os.path.join(self.labels_dir, file)
            with open(json_path, "r") as f:
                ann_data = json.load(f)
            img_info = {
                "id": i,
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
                    "image_id": i,
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
        try:
            with open(output_json_path, "w") as f:
                json.dump(coco_synthetic, f, indent=2)
            logging.info(f"Synthetic COCO JSON guardado en {output_json_path}")
        except Exception as e:
            logging.error(f"Error al guardar el JSON final: {e}")

        return synthetic_counts_by_class, synthetic_total
