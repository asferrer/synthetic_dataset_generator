# tests/test_class_analysis.py
import pytest
from src.analysis.class_analysis import analyze_coco_dataset

def test_analyze_coco_dataset():
    # Crear un dataset dummy en formato COCO.
    coco_data = {
        "images": [
            {"id": 1, "file_name": "img1.jpg"},
            {"id": 2, "file_name": "img2.jpg"}
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 50]},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [20, 20, 30, 30]},
            {"id": 3, "image_id": 2, "category_id": 1, "bbox": [15, 15, 40, 40]}
        ],
        "categories": [
            {"id": 1, "name": "cat"},
            {"id": 2, "name": "dog"}
        ]
    }
    
    analysis = analyze_coco_dataset(coco_data)
    assert analysis["num_images"] == 2
    assert analysis["num_annotations"] == 3
    # Se espera que "cat" tenga 2 instancias y "dog" 1.
    assert analysis["class_counts"]["cat"] == 2
    assert analysis["class_counts"]["dog"] == 1
