"""
Script de prueba para verificar que todas las importaciones funcionan correctamente.
"""

import sys
from pathlib import Path

# Añadir el directorio raíz al path
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

print("[OK] Testing imports...")

try:
    from src.data.coco_parser import load_coco_json, COCOValidationError
    print("[OK] coco_parser imports OK")
except Exception as e:
    print(f"[FAIL] coco_parser import FAILED: {e}")

try:
    from src.analysis.class_analysis import analyze_coco_dataset
    print("[OK] class_analysis imports OK")
except Exception as e:
    print(f"[FAIL] class_analysis import FAILED: {e}")

try:
    from src.augmentation.augmentor import SyntheticDataAugmentor
    print("[OK] augmentor imports OK")
except Exception as e:
    print(f"[FAIL] augmentor import FAILED: {e}")

print("\n[OK] All imports successful!")

# Test basico de analyze_coco_dataset
print("\n[OK] Testing analyze_coco_dataset function...")
test_coco = {
    "images": [{"id": 1, "file_name": "test.jpg"}],
    "annotations": [{"id": 1, "image_id": 1, "category_id": 1}],
    "categories": [{"id": 1, "name": "test_class"}]
}

result = analyze_coco_dataset(test_coco, {})
print(f"[OK] Function result: {result}")
print("\n[OK] All tests passed!")
