# src/data/coco_parser.py
import json
import os
import logging


class COCOValidationError(Exception):
    """Excepción personalizada para errores de validación de formato COCO."""
    pass


def validate_coco_structure(data, strict=True):
    """Valida exhaustivamente la estructura de un dataset en formato COCO.

    Args:
        data: Diccionario con datos COCO a validar
        strict: Si True, lanza excepciones; si False, solo muestra warnings

    Returns:
        bool: True si la validación es exitosa

    Raises:
        COCOValidationError: Si hay errores críticos de estructura (solo en modo strict)
    """
    errors = []
    warnings = []

    # ===== Validar claves principales =====
    required_keys = ['images', 'annotations', 'categories']
    for key in required_keys:
        if key not in data:
            errors.append(f"Clave requerida '{key}' no encontrada en el JSON")
        elif not isinstance(data[key], list):
            errors.append(f"'{key}' debe ser una lista, encontrado: {type(data[key])}")

    # Si faltan claves críticas, abortar validación
    if errors:
        error_msg = f"Errores críticos de estructura:\n  - " + "\n  - ".join(errors)
        if strict:
            raise COCOValidationError(error_msg)
        else:
            logging.error(error_msg)
            return False

    # ===== Validar categorías =====
    categories = data['categories']
    cat_ids = set()
    cat_names = set()

    if not categories:
        warnings.append("No se encontraron categorías en el dataset")

    for i, cat in enumerate(categories):
        # Validar campos requeridos
        if 'id' not in cat:
            errors.append(f"Categoría #{i} sin campo 'id': {cat}")
            continue
        if 'name' not in cat:
            errors.append(f"Categoría #{i} (id={cat.get('id')}) sin campo 'name'")
            continue

        # Validar unicidad de IDs
        if cat['id'] in cat_ids:
            errors.append(f"ID de categoría duplicado: {cat['id']}")
        cat_ids.add(cat['id'])

        # Advertir sobre nombres duplicados
        if cat['name'] in cat_names:
            warnings.append(f"Nombre de categoría duplicado: '{cat['name']}' "
                          f"(puede causar confusión)")
        cat_names.add(cat['name'])

    # ===== Validar imágenes =====
    images = data['images']
    image_ids = set()
    image_files = set()

    if not images:
        warnings.append("No se encontraron imágenes en el dataset")

    for i, img in enumerate(images):
        # Validar campos requeridos
        required_img_fields = ['id', 'file_name']
        for field in required_img_fields:
            if field not in img:
                errors.append(f"Imagen #{i} sin campo '{field}': {img}")
                continue

        # Validar unicidad de IDs
        if img.get('id') in image_ids:
            errors.append(f"ID de imagen duplicado: {img['id']}")
        else:
            image_ids.add(img['id'])

        # Advertir sobre nombres de archivo duplicados
        file_name = img.get('file_name')
        if file_name in image_files:
            warnings.append(f"Nombre de archivo duplicado: '{file_name}' "
                          f"(puede causar sobrescritura)")
        image_files.add(file_name)

        # Validar dimensiones si están presentes
        if 'width' in img and 'height' in img:
            if img['width'] <= 0 or img['height'] <= 0:
                errors.append(f"Imagen {img['id']} tiene dimensiones inválidas: "
                            f"{img['width']}x{img['height']}")

    # ===== Validar anotaciones =====
    annotations = data['annotations']
    ann_ids = set()
    ann_per_image = {}
    ann_per_category = {}

    for i, ann in enumerate(annotations):
        # Validar campos requeridos
        required_ann_fields = ['id', 'image_id', 'category_id']
        missing_fields = [f for f in required_ann_fields if f not in ann]
        if missing_fields:
            errors.append(f"Anotación #{i} sin campos: {missing_fields}")
            continue

        ann_id = ann['id']
        img_id = ann['image_id']
        cat_id = ann['category_id']

        # Validar unicidad de IDs
        if ann_id in ann_ids:
            errors.append(f"ID de anotación duplicado: {ann_id}")
        ann_ids.add(ann_id)

        # Verificar referencias válidas
        if img_id not in image_ids:
            errors.append(f"Anotación {ann_id} refiere a imagen inexistente: {img_id}")

        if cat_id not in cat_ids:
            errors.append(f"Anotación {ann_id} refiere a categoría inexistente: {cat_id}")

        # Estadísticas para reporte
        ann_per_image[img_id] = ann_per_image.get(img_id, 0) + 1
        ann_per_category[cat_id] = ann_per_category.get(cat_id, 0) + 1

        # Validar bbox si está presente
        if 'bbox' in ann:
            bbox = ann['bbox']
            if not isinstance(bbox, list) or len(bbox) != 4:
                errors.append(f"Anotación {ann_id} tiene bbox inválido: {bbox}")
            elif any(v < 0 for v in bbox):
                warnings.append(f"Anotación {ann_id} tiene valores negativos en bbox")

        # Validar área si está presente
        if 'area' in ann:
            if ann['area'] <= 0:
                warnings.append(f"Anotación {ann_id} tiene área <= 0: {ann['area']}")

    # ===== Reportar errores y warnings =====
    if errors:
        error_msg = f"Se encontraron {len(errors)} errores de validación:\n"
        error_msg += "\n".join(f"  {i+1}. {err}" for i, err in enumerate(errors[:20]))
        if len(errors) > 20:
            error_msg += f"\n  ... y {len(errors) - 20} errores más"

        if strict:
            raise COCOValidationError(error_msg)
        else:
            logging.error(error_msg)
            return False

    if warnings:
        warning_msg = f"Se encontraron {len(warnings)} advertencias:\n"
        warning_msg += "\n".join(f"  - {warn}" for warn in warnings[:10])
        if len(warnings) > 10:
            warning_msg += f"\n  ... y {len(warnings) - 10} advertencias más"
        logging.warning(warning_msg)

    # ===== Reporte de estadísticas =====
    logging.info(f"✓ Validación COCO exitosa:")
    logging.info(f"  - {len(image_ids)} imágenes")
    logging.info(f"  - {len(cat_ids)} categorías: {sorted(cat_names)}")
    logging.info(f"  - {len(ann_ids)} anotaciones")

    if ann_per_category:
        logging.info(f"  - Distribución por categoría:")
        for cat_id, count in sorted(ann_per_category.items()):
            cat_name = next((c['name'] for c in categories if c['id'] == cat_id), 'unknown')
            logging.info(f"    • {cat_name}: {count} anotaciones")

    return True


def load_coco_json(json_path, validate=True, strict=True):
    """Carga y valida un archivo JSON en formato COCO con manejo robusto de errores.

    Args:
        json_path: Ruta al archivo JSON
        validate: Si True, valida la estructura COCO
        strict: Si True, lanza excepciones en errores; si False, solo warnings

    Returns:
        Diccionario con los datos del dataset

    Raises:
        FileNotFoundError: Si el archivo no existe
        COCOValidationError: Si la estructura es inválida (solo en modo strict)
        ValueError: Si el JSON es inválido
    """
    # ===== Validar que el archivo existe =====
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Archivo COCO no encontrado: {json_path}")

    if not os.path.isfile(json_path):
        raise ValueError(f"La ruta no es un archivo: {json_path}")

    # ===== Cargar JSON con manejo de errores =====
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON inválido en {json_path}:\n  Línea {e.lineno}, "
                        f"columna {e.colno}: {e.msg}")
    except UnicodeDecodeError as e:
        raise ValueError(f"Error de codificación en {json_path}: {e}")
    except Exception as e:
        raise IOError(f"Error leyendo {json_path}: {e}")

    # ===== Validar estructura si está habilitado =====
    if validate:
        validate_coco_structure(data, strict=strict)

    return data

def get_category_mapping(coco_data):
    """
    Genera un diccionario que mapea el id de cada categoría a su nombre.
    
    Parámetros:
      - coco_data: Diccionario del dataset COCO.
    
    Retorna:
      - Dict { category_id: category_name, ... }
    """
    return {cat["id"]: cat["name"] for cat in coco_data.get("categories", [])}
