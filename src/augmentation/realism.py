import cv2
import numpy as np
import logging

def apply_poisson_blending(src_obj, background, mask, center):
    """Aplica Poisson Blending para clonar un objeto de forma suave."""
    try:
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        return cv2.seamlessClone(src_obj, background, mask, center, cv2.NORMAL_CLONE)
    except cv2.error as e:
        logging.error(f"Error durante Poisson Blending: {e}. Se usará pegado simple.")
        inv_mask = cv2.bitwise_not(mask)
        fg = cv2.bitwise_and(src_obj, src_obj, mask=mask)
        bg = cv2.bitwise_and(background, background, mask=inv_mask)
        return cv2.add(bg, fg)

def transfer_color_correction(obj_img, bg_roi, mask, intensity=0.7):
    """
    Ajusta el color del objeto al del fondo y lo mezcla con el original
    para un efecto más sutil.
    """
    try:
        if cv2.countNonZero(mask) == 0:
            return obj_img

        obj_lab = cv2.cvtColor(obj_img, cv2.COLOR_BGR2LAB)
        bg_lab = cv2.cvtColor(bg_roi, cv2.COLOR_BGR2LAB)

        mean_bg, std_bg = cv2.meanStdDev(bg_lab, mask=mask)
        mean_obj, std_obj = cv2.meanStdDev(obj_lab, mask=mask)

        std_obj[std_obj == 0] = 1

        l, a, b = cv2.split(obj_lab)
        l = np.clip((l - mean_obj[0]) * (std_bg[0] / std_obj[0]) + mean_bg[0], 0, 255).astype(np.uint8)
        a = np.clip((a - mean_obj[1]) * (std_bg[1] / std_obj[1]) + mean_bg[1], 0, 255).astype(np.uint8)
        b = np.clip((b - mean_obj[2]) * (std_bg[2] / std_obj[2]) + mean_bg[2], 0, 255).astype(np.uint8)

        corrected_lab = cv2.merge([l, a, b])
        corrected_bgr = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)
        
        # Mezclar el resultado con el original para un efecto más suave
        blended_bgr = cv2.addWeighted(obj_img, 1 - intensity, corrected_bgr, intensity, 0)

        return cv2.bitwise_and(blended_bgr, blended_bgr, mask=mask)
    except Exception as e:
        logging.error(f"Error en transfer_color_correction: {e}")
        return obj_img

def estimate_blur(image):
    """Estima el nivel de desenfoque de una imagen usando la varianza del Laplaciano."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(image, cv2.CV_64F).var()

def match_blur(obj_img, bg_roi, mask):
    """Ajusta el desenfoque del objeto para que coincida con el del fondo."""
    try:
        if cv2.countNonZero(mask) == 0:
            return obj_img
            
        bg_blur = estimate_blur(bg_roi)
        obj_blur = estimate_blur(obj_img)

        if obj_blur > bg_blur * 1.5:
            kernel_size = int((obj_blur / bg_blur) * 0.5) * 2 + 1
            kernel_size = min(kernel_size, 21)
            blurred_obj = cv2.GaussianBlur(obj_img, (kernel_size, kernel_size), 0)
            return cv2.bitwise_and(blurred_obj, blurred_obj, mask=mask)
        return obj_img
    except Exception as e:
        logging.error(f"Error en match_blur: {e}")
        return obj_img

def generate_shadow(mask, strength=0.7, blur_kernel_size=21, offset_x=10, offset_y=10):
    """Genera una sombra suave y proyectada a partir de la máscara de un objeto."""
    shadow_layer = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    shadow_layer[mask == 255] = (0, 0, 0, int(255 * strength))
    shadow_layer = cv2.GaussianBlur(shadow_layer, (blur_kernel_size, blur_kernel_size), 0)
    M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
    return cv2.warpAffine(shadow_layer, M, (shadow_layer.shape[1], shadow_layer.shape[0]))

def add_lighting_effect(obj_img, light_type='spotlight', strength=1.5):
    """Añade un efecto de iluminación simple al objeto."""
    h, w = obj_img.shape[:2]
    light_map = np.zeros((h, w), dtype=np.float32)

    if light_type == 'spotlight':
        center_x, center_y = w // 2, h // 2
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        radius = max(h, w) / 2.0
        light_map = np.exp(-(dist_from_center / radius)**2)
        light_map = (light_map - np.min(light_map)) / (np.max(light_map) - np.min(light_map))
        light_map *= strength

    hls = cv2.cvtColor(obj_img, cv2.COLOR_BGR2HLS).astype(np.float32)
    hls[:, :, 1] *= light_map
    hls[:, :, 1] = np.clip(hls[:, :, 1], 0, 255)
    return cv2.cvtColor(hls.astype(np.uint8), cv2.COLOR_HLS2BGR)

def apply_underwater_effect(obj_img, color_cast=(120, 80, 20), intensity=0.6):
    """
    Simula el efecto del agua sobre un objeto, añadiendo un velo de color y reduciendo el contraste.
    """
    try:
        haze_color = np.full(obj_img.shape, color_cast, dtype=obj_img.dtype)
        hazed_obj = cv2.addWeighted(obj_img, 1 - intensity, haze_color, intensity, 0)
        return hazed_obj
    except Exception as e:
        logging.error(f"Error en apply_underwater_effect: {e}")
        return obj_img

def add_upscaling_noise(image, intensity=10):
    """Añade ruido Gaussiano a una imagen para simular artefactos de escalado."""
    h, w, c = image.shape
    noise = np.random.randn(h, w, c) * intensity
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image

def laplacian_pyramid_blending(obj_img, background, mask, num_levels=4):
    """
    Fusiona una imagen de objeto en un fondo usando pirámides laplacianas.
    
    Parámetros:
      - obj_img: Imagen del objeto a pegar (BGR).
      - background: Imagen de fondo (BGR).
      - mask: Máscara binaria del objeto (un solo canal).
      - num_levels: Número de niveles en la pirámide.
    
    Retorna:
      - Imagen fusionada.
    """
    # Generar pirámide Gaussiana para el objeto, el fondo y la máscara
    gp_obj = [obj_img.astype(np.float32)]
    gp_bg = [background.astype(np.float32)]
    gp_mask = [mask.astype(np.float32) / 255.0]

    for i in range(num_levels):
        gp_obj.append(cv2.pyrDown(gp_obj[i]))
        gp_bg.append(cv2.pyrDown(gp_bg[i]))
        gp_mask.append(cv2.pyrDown(gp_mask[i]))

    # Generar pirámide Laplaciana para el objeto y el fondo
    lp_obj = [gp_obj[num_levels-1]]
    lp_bg = [gp_bg[num_levels-1]]
    for i in range(num_levels - 1, 0, -1):
        # Para el objeto
        size = (gp_obj[i-1].shape[1], gp_obj[i-1].shape[0])
        expanded_obj = cv2.pyrUp(gp_obj[i], dstsize=size)
        lp_obj.append(cv2.subtract(gp_obj[i-1], expanded_obj))
        # Para el fondo
        size = (gp_bg[i-1].shape[1], gp_bg[i-1].shape[0])
        expanded_bg = cv2.pyrUp(gp_bg[i], dstsize=size)
        lp_bg.append(cv2.subtract(gp_bg[i-1], expanded_bg))
        
    # Fusionar cada nivel de la pirámide
    lp_fused = []
    for l_obj, l_bg, g_mask in zip(lp_obj, lp_bg, gp_mask[::-1]):
        # Asegurar que la máscara tenga 3 canales para la fusión
        g_mask_3c = cv2.merge([g_mask, g_mask, g_mask])
        fused_level = l_obj * g_mask_3c + l_bg * (1 - g_mask_3c)
        lp_fused.append(fused_level)

    # Reconstruir la imagen final
    fused_reconstruction = lp_fused[0]
    for i in range(1, len(lp_fused)):
        size = (lp_fused[i].shape[1], lp_fused[i].shape[0])
        fused_reconstruction = cv2.pyrUp(fused_reconstruction, dstsize=size)
        fused_reconstruction = cv2.add(fused_reconstruction, lp_fused[i])
        
    return np.clip(fused_reconstruction, 0, 255).astype(np.uint8)

def generate_caustics_map(width, height, complexity=0.01, brightness=0.4):
    """
    Genera un mapa de cáusticas procedural utilizando ruido Perlin.
    """
    import noise
    
    caustics_map = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            # Generar ruido con varias octavas para un patrón más natural
            value = noise.pnoise2(x * complexity, y * complexity, octaves=4)
            caustics_map[y, x] = value

    # Normalizar y ajustar el brillo
    caustics_map = (caustics_map - np.min(caustics_map)) / (np.max(caustics_map) - np.min(caustics_map))
    caustics_map = np.clip(caustics_map * brightness + (1 - brightness), 0, 2) # Permite zonas más brillantes que el original
    
    return cv2.cvtColor(caustics_map, cv2.COLOR_GRAY2BGR)

def apply_caustics(image, caustics_map):
    """
    Aplica el mapa de cáusticas a una imagen.
    """
    # Convertir a HLS para manipular la luminosidad sin alterar el color
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS).astype(np.float32)
    hls[:, :, 1] *= caustics_map[:, :, 1] # Aplicar a la luminosidad
    hls[:, :, 1] = np.clip(hls[:, :, 1], 0, 255)
    
    return cv2.cvtColor(hls.astype(np.uint8), cv2.COLOR_HLS2BGR)