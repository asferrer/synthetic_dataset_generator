# src/visualization/visualizer.py
import matplotlib.pyplot as plt

def plot_class_distribution(class_counts):
    """
    Genera y retorna una figura de matplotlib con la distribución de clases.
    
    Parámetros:
      - class_counts: Diccionario { clase: cantidad, ... }
    
    Retorna:
      - Figura de matplotlib.
    """
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(classes, counts, color='skyblue')
    ax.set_xlabel("Clases")
    ax.set_ylabel("Número de instancias")
    ax.set_title("Distribución de clases en el dataset")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig
