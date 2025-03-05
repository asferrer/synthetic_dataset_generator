# src/utils/helpers.py
import os

def ensure_dir(directory):
    """
    Crea el directorio si no existe.
    
    Parámetros:
      - directory: Ruta del directorio a crear.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
