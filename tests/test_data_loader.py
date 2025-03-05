# tests/test_data_loader.py
import os
import cv2
import numpy as np
import pytest
from src.data.dataset_loader import load_images_from_folder

def create_dummy_image_file(directory, filename="dummy.jpg", width=100, height=100, channels=3):
    """Crea y guarda una imagen dummy en el directorio especificado."""
    dummy_image = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
    file_path = os.path.join(directory, filename)
    cv2.imwrite(file_path, dummy_image)
    return file_path

def test_load_images_from_folder(tmp_path):
    # tmp_path es un objeto pathlib.Path proporcionado por pytest para directorios temporales.
    tmp_dir = tmp_path / "images"
    tmp_dir.mkdir()
    # Crear 3 imágenes dummy.
    for i in range(3):
        create_dummy_image_file(str(tmp_dir), filename=f"dummy_{i}.jpg")
    
    images = load_images_from_folder(str(tmp_dir))
    assert len(images) == 3, "Debe cargarse 3 imágenes."
    
    # Crear un archivo no imagen y verificar que se ignora.
    with open(tmp_dir / "not_image.txt", "w") as f:
        f.write("No es una imagen.")
    images = load_images_from_folder(str(tmp_dir))
    # El número de imágenes debe seguir siendo 3.
    assert len(images) == 3
