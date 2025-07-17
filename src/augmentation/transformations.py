import cv2
import numpy as np
from scipy import ndimage

def rotate_image(image, angle):
    """Rota la imagen manteniendo su tamaño original y manejando el canal alfa."""
    return ndimage.rotate(image, angle, reshape=True, cval=0)

def scale_image(image, scale_factor):
    """Escala la imagen por el factor especificado."""
    if scale_factor == 1.0:
        return image
    height, width = image.shape[:2]
    new_dimensions = (int(width * scale_factor), int(height * scale_factor))
    return cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)

def apply_perspective_transform(image, magnitude=0.08):
    """Aplica una ligera transformación de perspectiva aleatoria."""
    h, w = image.shape[:2]
    
    # Puntos originales (esquinas)
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    
    # Puntos de destino con un desplazamiento aleatorio
    dw = w * magnitude
    dh = h * magnitude
    pts2 = np.float32([
        [np.random.uniform(-dw, dw), np.random.uniform(-dh, dh)],
        [w - np.random.uniform(-dw, dw), np.random.uniform(-dh, dh)],
        [np.random.uniform(-dw, dw), h - np.random.uniform(-dh, dh)],
        [w - np.random.uniform(-dw, dw), h - np.random.uniform(-dh, dh)]
    ])
    
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(image, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

def apply_motion_blur(image, kernel_size=15, angle=45):
    """Aplica un desenfoque de movimiento lineal."""
    kernel = np.zeros((kernel_size, kernel_size))
    
    # Crear una línea en el kernel
    center = kernel_size // 2
    radian_angle = np.deg2rad(angle)
    dx = np.cos(radian_angle)
    dy = np.sin(radian_angle)
    
    # Dibujar la línea en el kernel
    for i in range(kernel_size):
        x = int(center + (i - center) * dx)
        y = int(center + (i - center) * dy)
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1.0
            
    # Normalizar el kernel
    kernel /= np.sum(kernel)
    
    return cv2.filter2D(image, -1, kernel)