# src/analysis/class_analysis.py

def analyze_coco_dataset(coco_data):
    """
    Analiza el dataset en formato COCO y devuelve estadísticas básicas:
      - Número de imágenes.
      - Número de anotaciones.
      - Distribución de clases (cantidad de instancias por clase).
    
    Parámetros:
      - coco_data: Diccionario del dataset COCO.
    
    Retorna:
      - Diccionario con claves: num_images, num_annotations, class_counts.
    """
    images = coco_data.get("images", [])
    annotations = coco_data.get("annotations", [])
    categories = coco_data.get("categories", [])
    
    num_images = len(images)
    num_annotations = len(annotations)
    
    # Inicializar conteo de instancias por categoría
    class_counts = {cat["name"]: 0 for cat in categories}
    # Asumir que cada anotación tiene "category_id"
    cat_mapping = {cat["id"]: cat["name"] for cat in categories}
    
    for ann in annotations:
        cat_id = ann.get("category_id")
        cat_name = cat_mapping.get(cat_id, "desconocido")
        class_counts[cat_name] += 1
    
    return {
        "num_images": num_images,
        "num_annotations": num_annotations,
        "class_counts": class_counts
    }
