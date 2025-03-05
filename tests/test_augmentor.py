# tests/test_augmentor.py
import os
import cv2
import numpy as np
import tempfile
import pytest
from src.augmentation.augmentor import SyntheticDataAugmentor
from src.augmentation.transformations import rotate_image, scale_image, translate_image

def create_dummy_image(width=100, height=100, channels=3):
    """Genera una imagen aleatoria dummy."""
    return np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)

def test_rotate_point():
    augmentor = SyntheticDataAugmentor()
    point = [50, 50]
    origin = [50, 50]
    angle = 90
    # Rotar un punto respecto a sí mismo debe dar el mismo punto.
    new_point = augmentor.rotate_point(point, angle, origin)
    assert np.allclose(new_point, point), "Rotar un punto sobre sí mismo debe mantener sus coordenadas."
    
    # Caso de prueba: Rotar el punto (100, 50) 90° alrededor de (50, 50)
    point = [100, 50]
    new_point = augmentor.rotate_point(point, 90, origin)
    expected = [50, 100]  # Aproximación esperada
    assert np.allclose(new_point, expected, atol=1), f"Se esperaba {expected}, pero se obtuvo {new_point}"

def test_apply_transformations():
    augmentor = SyntheticDataAugmentor(rot=True, scale=True, trans=True, try_count=1)
    dummy_patch = create_dummy_image(50, 50)
    bg_shape = (200, 200, 3)
    
    transformed_patch, angle, scale_factor, pos = augmentor.apply_transformations(dummy_patch, bg_shape)
    
    # Verificar que la imagen transformada sea un array de numpy y no esté vacía
    assert isinstance(transformed_patch, np.ndarray)
    assert transformed_patch.size > 0
    
    # Comprobar que el ángulo esté en el rango [0, 360)
    assert 0 <= angle < 360
    
    # Verificar que el factor de escala sea positivo
    assert scale_factor > 0
    
    # Verificar que la posición esté dentro de las dimensiones del fondo
    x, y = pos
    assert 0 <= x < bg_shape[1]
    assert 0 <= y < bg_shape[0]

def test_paste_object():
    augmentor = SyntheticDataAugmentor(rot=True, scale=True, trans=True, try_count=3)
    bg_image = create_dummy_image(200, 200)
    bg_bin = np.zeros_like(bg_image, dtype=np.uint8)
    dummy_patch = create_dummy_image(50, 50)
    
    bg_bin_updated, bg_image_updated, angle, pos, scale_factor, patched = augmentor.paste_object(
        bg_bin, bg_image, dummy_patch, idx_obj=1
    )
    assert patched, "El patch debería pegarse exitosamente en el fondo."
    # Verificar que la imagen resultante tenga las mismas dimensiones que el fondo original.
    assert bg_image_updated.shape == bg_image.shape
