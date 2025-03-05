# Synthetic Data Augmentation Tool

![Python](https://img.shields.io/badge/Python-3.9-blue)
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
This tool is designed to generate synthetic data from an annotated image dataset (in COCO format) using data augmentation techniques. It allows you to either extract objects from the original images (using segmentation annotations) or use object images from an external folder, and then paste them on various backgrounds. The tool applies transformations (rotation, scaling, translation) and advanced strategies to adapt the object's tone and smooth its edges, ensuring a natural blend between the object and the background.


![Algorithm example](/assets/tool1.png)
![Algorithm definition](/assets/tool2.png)

Moreover, the tool controls the overlapping between objects using the actual object contour (beyond simple bounding boxes) and lets you specify the desired number of synthetic samples per class via the user interface to balance the dataset. At the end of the process, it generates a single JSON file in COCO format that aggregates all the annotations of the new synthetic dataset.


## Features

- **Segmented Object Extraction:**  
  Extract objects from the original images based on segmentation annotations in the input dataset.

- **External Object Usage:**  
  Use object images stored in an external folder, organized in subdirectories by class.

- **Transformations and Adaptation:**  
  Apply random transformations (rotation, scaling, translation) and adjust the object’s tone and edges so that it blends seamlessly with the background.

- **Overlap Control:**  
  Employ a strategy based on the object's mask (its actual contour) with dilation to avoid excessive overlapping of objects.

- **Custom Configuration:**  
  The Streamlit-based user interface allows you to:
  - Select the object source (input dataset or external folder).
  - Define the maximum number of objects to paste per image.
  - Specify the desired number of synthetic samples per class.
  - View suggestions based on the original dataset analysis.
  - Monitor the process progress with a progress bar.
  - View graphs showing the number of synthetic samples generated per class and the final dataset composition (original + synthetic).

- **COCO Format Output:**  
  A single global JSON file is generated in COCO format with the same structure as the input JSON. It includes:
  - Information on the generated synthetic images.
  - Annotations (bounding boxes, areas) of the pasted objects.
  - Category information.

## Installation

1. **Clone the repository:**
   ```
   git clone https://github.com/tu_usuario/synthetic-data-augmentation.git
   cd synthetic-data-augmentation
   ```

2. Install dependencies (using a virtual environment is recommended):
  ```
  pip install -r requirements.txt
  ```

3. (Optional) Build the Docker image:
  ```
  docker-compose build
  ```

# Usage
## With Docker
1. Start the application:
  ```
  docker-compose up
  ```

2. Access the Streamlit interface at http://localhost:8501.

## Without Docker
1. Run the application:
  ```
  streamlit run app/main.py
  ```

2. Use the interface to:

  - Upload your COCO JSON file.
  - Select the classes to augment.
  - Define the desired number of synthetic samples per class (with suggestions based on the dataset).
  - Choose the object source and configure transformation parameters.
  - Run the augmentation process while monitoring progress and viewing output graphs.

# Customization
- **Configuration**:
Modify the file ```configs/config.yaml``` to adjust paths, augmentation parameters (rotation, scaling, etc.), and other global settings.

- **Overlap and Transformation Parameters:**
Parameters such as ```overlap_threshold``` and ```max_overlap_threshold``` can be adjusted in the code to achieve optimal blending between objects and backgrounds.

- **Desired Sample Control:**
Through the Streamlit interface, you can define the exact number of synthetic samples to generate per class, overriding the automatic balancing based on the dataset.

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