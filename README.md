# Synthetic Data Augmentation Tool

![Python](https://img.shields.io/badge/Python-3.10-blue)
![License](https://img.shields.io/badge/License-Apache%202.0-green)
<a href="mailto:your.email@example.com">
    <img alt="email" src="https://img.shields.io/badge/contact_me-email-yellow">
</a>

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [With Docker](#with-docker)
  - [Without Docker](#without-docker)
- [Customization](#customization)
- [Citing](#citing)
- [Contact](#contact)

## Project Overview
Esta herramienta está diseñada para generar datos sintéticos a partir de un dataset de imágenes anotadas (en formato COCO) mediante técnicas de aumento de datos. La herramienta permite extraer objetos de las imágenes originales (usando segmentaciones) o utilizar imágenes de objetos desde una carpeta externa, y pegarlos sobre fondos, aplicando transformaciones (rotación, escalado y traslación) y estrategias avanzadas para adaptar la tonalidad y suavizar los bordes, logrando una integración natural entre el objeto y el fondo.

Además, la herramienta controla el solapamiento entre objetos utilizando el contorno real (más allá de las bounding boxes) y permite definir desde la interfaz el número deseado de muestras sintéticas por clase para balancear el dataset. Al finalizar el proceso, se genera un archivo JSON en formato COCO que agrupa todas las anotaciones del nuevo dataset sintético.

## Features

- **Extracción de objetos segmentados:**  
  Extrae objetos de las imágenes originales a partir de las anotaciones de segmentación del dataset de entrada.

- **Uso de objetos externos:**  
  Permite utilizar imágenes de objetos almacenadas en una carpeta externa, organizadas en subdirectorios por clase.

- **Transformaciones y adaptación:**  
  Aplica transformaciones aleatorias (rotación, escalado, traslación) y adapta el tono y el contorno del objeto para que se integre mejor con el fondo.

- **Control del solapamiento:**  
  Emplea una estrategia basada en la máscara (contorno real) del objeto, con dilatación, para evitar que los objetos se solapen de forma excesiva.

- **Configuración personalizada:**  
  La interfaz de usuario basada en Streamlit permite:
  - Seleccionar la fuente de objetos (Dataset de entrada o Carpeta de objetos).
  - Definir el número máximo de objetos a pegar por imagen.
  - Especificar el número deseado de muestras sintéticas por cada clase.
  - Visualizar sugerencias basadas en el análisis del dataset original.
  - Monitorizar el progreso del proceso mediante una barra de progreso.
  - Visualizar gráficos del número de muestras sintéticas generadas por clase y la composición final del dataset (original + sintéticas).

- **Salida en formato COCO:**  
  Se genera un único archivo JSON global en formato COCO con la misma estructura del JSON de entrada, que incluye:
  - Información de las imágenes sintéticas generadas.
  - Anotaciones (bounding boxes, áreas) de los objetos pegados.
  - Información de las categorías.

## Installation

1. Clona el repositorio:
   ```
   git clone https://github.com/tu_usuario/synthetic-data-augmentation.git
   cd synthetic-data-augmentation
   ```

2. Instala las dependencias (se recomienda usar un entorno virtual):
  ```
  pip install -r requirements.txt
  ```

3. (Opcional) Construye la imagen Docker:
  ```
  docker-compose build
  ```

# Usage
## With Docker
1. Levanta la aplicación:
  ```
  docker-compose up
  ```

2. Accede a la interfaz de Streamlit en **http://localhost:8501**.

## Without Docker
1. Ejecuta la aplicación:
  ```
  streamlit run app/main.py
  ```

2. Utiliza la interfaz para:

- Subir tu archivo COCO JSON.
- Seleccionar las clases a aumentar.
- Definir el número deseado de muestras sintéticas por clase (con sugerencias).
- Elegir la fuente de objetos y configurar parámetros de transformación.
- Ejecutar el proceso y visualizar el progreso y las gráficas de salida.

# Customization
- **Configuración**:
Modifica el archivo ```configs/config.yaml``` para ajustar rutas, parámetros de aumento (rotación, escalado, etc.) y otros ajustes globales.

- **Parámetros de Overlap y Transformaciones:**
Los parámetros como ```overlap_threshold``` y ```max_overlap_threshold``` en el código pueden ajustarse para lograr una integración óptima entre los objetos y los fondos.

## Citing
If you use this repo in your research/project or wish to refer to the results published here, please use the following BibTeX entries.

```BibTeX
@article{SANCHEZFERRER2023154,
      title = {An experimental study on marine debris location and recognition using object detection},
      author = {Alejandro Sánchez-Ferrer and Jose J. Valero-Mas and Antonio Javier Gallego and Jorge Calvo-Zaragoza},
      journal = {Pattern Recognition Letters},
      year = {2023},
      doi = {https://doi.org/10.1016/j.patrec.2022.12.019},
      url = {https://www.sciencedirect.com/science/article/pii/S0167865522003889},
}
```
```BibTeX
@InProceedings{10.1007/978-3-031-04881-4_49,
      title="The CleanSea Set: A Benchmark Corpus for Underwater Debris Detection and Recognition",
      author="S{\'a}nchez-Ferrer, Alejandro and Gallego, Antonio Javier and Valero-Mas, Jose J. and Calvo-Zaragoza, Jorge",
      booktitle="Pattern Recognition and Image Analysis",
      year="2022",
      publisher="Springer International Publishing",
}
```

# Contact
**Project Lead:** Alejandro Sanchez Ferrer  
**Email:** asanc.tech@gmail.com  
**GitHub:** [asferrer](https://github.com/asferrer)