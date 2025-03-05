# src/data/coco_parser.py
import json

def load_coco_json(json_path):
    """
    Carga y devuelve el contenido de un archivo JSON en formato COCO.
    
    Parámetros:
      - json_path: Ruta al archivo JSON.
    
    Retorna:
      - Diccionario con los datos del dataset.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
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
