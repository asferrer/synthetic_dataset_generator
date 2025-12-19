def analyze_coco_dataset(coco_data, class_mapping=None):
    """
    Analiza el dataset en formato COCO y devuelve estadísticas básicas.
    Si se proporciona un class_mapping, agrupa las clases de origen en la clase objetivo.
    """
    if class_mapping is None:
        class_mapping = {}

    images = coco_data.get("images", [])
    annotations = coco_data.get("annotations", [])
    categories = coco_data.get("categories", [])
    
    num_images = len(images)
    num_annotations = len(annotations)
    
    # Conteo inicial de instancias por categoría original
    original_class_counts = {cat["name"]: 0 for cat in categories}
    cat_id_to_name = {cat["id"]: cat["name"] for cat in categories}
    
    for ann in annotations:
        cat_id = ann.get("category_id")
        cat_name = cat_id_to_name.get(cat_id)
        if cat_name:
            original_class_counts[cat_name] += 1
    
    # Procesar la agrupación de clases
    grouped_class_counts = original_class_counts.copy()
    processed_sources = set()

    for target_class, source_classes in class_mapping.items():
        # Sumar las cuentas de las clases de origen
        grouped_count = sum(original_class_counts.get(src, 0) for src in source_classes)
        
        # Añadir la nueva clase agrupada
        grouped_class_counts[target_class] = grouped_count
        
        # Marcar las clases de origen como procesadas para eliminarlas después
        for src in source_classes:
            processed_sources.add(src)

    # Eliminar las clases de origen que han sido agrupadas
    final_class_counts = {cls: count for cls, count in grouped_class_counts.items() if cls not in processed_sources}

    return {
        "num_images": num_images,
        "num_annotations": num_annotations,
        "class_counts": final_class_counts
    }