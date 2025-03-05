# src/data/dataset_loader.py
import os
import cv2
from tqdm import tqdm

def load_images_from_folder(folder_path, valid_exts=(".png", ".jpg", ".jpeg")):
    """
    Carga todas las imágenes de un directorio dado.
    
    Parámetros:
      - folder_path: Ruta al directorio.
      - valid_exts: Tupla con las extensiones válidas.
    
    Retorna:
      - Lista de imágenes cargadas con cv2.
    """
    images = []
    for filename in tqdm(os.listdir(folder_path), desc="Cargando imágenes"):
        if filename.lower().endswith(valid_exts):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
    return images
