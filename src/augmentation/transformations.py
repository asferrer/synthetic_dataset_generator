# src/augmentation/transformations.py
import cv2
import numpy as np
from scipy import ndimage

def rotate_image(image, angle):
    """
    Rota la imagen en el ángulo especificado (en grados) utilizando ndimage.
    """
    rotated = ndimage.rotate(image, angle, reshape=True)
    return rotated

def scale_image(image, scale_factor):
    """
    Escala la imagen por el factor especificado.
    """
    height, width = image.shape[:2]
    new_dimensions = (int(width * scale_factor), int(height * scale_factor))
    scaled = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)
    return scaled

def translate_image(image, x_offset, y_offset):
    """
    Traslada la imagen según los offsets en x e y.
    """
    height, width = image.shape[:2]
    M = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
    translated = cv2.warpAffine(image, M, (width, height))
    return translated
