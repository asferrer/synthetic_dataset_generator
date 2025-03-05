# app/main.py

import os
import json
import streamlit as st
import yaml
import matplotlib.pyplot as plt
import numpy as np

from src.data.coco_parser import load_coco_json
from src.analysis.class_analysis import analyze_coco_dataset
from src.visualization.visualizer import plot_class_distribution
from src.augmentation.augmentor import SyntheticDataAugmentor

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = load_config()

default_images_path = config["paths"]["backgrounds_dataset"]  # Carpeta de fondos
default_output_dir = os.path.dirname(config["paths"]["images"])  # Carpeta de salida sintética
default_objects_path = config["paths"].get("objects_dataset", None)  # Carpeta de objetos externos

def plot_synthetic_counts(synthetic_counts):
    classes = list(synthetic_counts.keys())
    counts = list(synthetic_counts.values())
    fig, ax = plt.subplots()
    ax.bar(classes, counts, color='orange')
    ax.set_title("Muestras sintéticas generadas por clase")
    ax.set_xlabel("Clase")
    ax.set_ylabel("Número de instancias sintéticas")
    plt.xticks(rotation=45, ha='right')
    return fig

def plot_final_composition(original_counts, synthetic_counts):
    final = {}
    for cls in set(original_counts.keys()).union(synthetic_counts.keys()):
        final[cls] = original_counts.get(cls, 0) + synthetic_counts.get(cls, 0)
    classes = list(final.keys())
    counts = list(final.values())
    fig, ax = plt.subplots()
    ax.bar(classes, counts, color='blue')
    ax.set_title("Composición final del dataset (original + sintéticas)")
    ax.set_xlabel("Clase")
    ax.set_ylabel("Número de instancias")
    plt.xticks(rotation=45, ha='right')
    return fig

def main():
    st.title("Herramienta de Aumento de Datos Sintético")
    
    st.sidebar.header("Configuración de Aumentado de Datos")
    rot = st.sidebar.checkbox("Rotación", value=config["augmentation"]["rot"])
    scale = st.sidebar.checkbox("Escalado", value=config["augmentation"]["scale"])
    trans = st.sidebar.checkbox("Traslación", value=config["augmentation"]["trans"])
    try_count = st.sidebar.number_input("Intentos máximos de pegado", min_value=1, max_value=10, value=config["augmentation"]["try_count"])
    overlap_threshold = st.sidebar.number_input("Umbral de solapamiento (%)", min_value=0, max_value=100, value=config["augmentation"]["overlap_threshold"])
    minority_threshold = st.sidebar.number_input("Umbral para clase minoritaria (número de instancias)", min_value=1, value=10)
    
    max_obj_per_image = st.sidebar.number_input("Máximo de objetos por imagen", min_value=1, value=config["augmentation"].get("max_objects_per_image", 3))
    
    st.sidebar.subheader("Fuente de objetos para aumentación")
    objects_source = st.sidebar.radio("Selecciona la fuente de objetos:", options=["Dataset de entrada", "Carpeta de objetos"])
    
    mode_bbox = st.sidebar.checkbox("Usar modo segmentación/bounding box (objetos y fondos)", value=True)
    
    st.header("Carga y Análisis del Dataset COCO")
    st.markdown("Sube tu archivo de anotaciones en formato COCO JSON.")
    
    coco_file = st.file_uploader("Selecciona tu archivo COCO JSON", type=["json"])
    
    if coco_file is not None:
        try:
            coco_data = json.load(coco_file)
            st.success("Archivo COCO cargado correctamente.")
            
            analysis = analyze_coco_dataset(coco_data)
            st.subheader("Resumen del Dataset Original")
            st.write(f"Número de imágenes: {analysis.get('num_images', 'N/A')}")
            st.write(f"Número de anotaciones: {analysis.get('num_annotations', 'N/A')}")
            original_counts = analysis.get("class_counts", {})
            st.write("Distribución de clases (original):")
            st.write(original_counts)
            
            fig_orig = plot_class_distribution(original_counts)
            st.pyplot(fig_orig)
            
            # Sugerir cantidad de muestras sintéticas para cada clase:
            if original_counts:
                target_count = max(original_counts.values())
                suggestion = {cls: max(target_count - original_counts.get(cls, 0), 0) for cls in original_counts.keys()}
                st.subheader("Sugerencia de muestras sintéticas por clase")
                st.write("Se recomienda generar estas muestras sintéticas para balancear el dataset:")
                st.write(suggestion)
            else:
                suggestion = {}

            minority_classes = [cls for cls, count in original_counts.items() if count < minority_threshold]
            st.write(f"Clases minoritarias (menos de {minority_threshold} instancias):")
            st.write(minority_classes)
            
            selected_classes = st.multiselect("Selecciona las clases a aumentar", minority_classes)
            
            # Para cada clase seleccionada, permitir definir manualmente el número de muestras sintéticas deseadas,
            # prellenando el campo con la sugerencia calculada.
            desired_samples = {}
            if selected_classes:
                st.subheader("Definir número de muestras sintéticas por clase")
                for cls in selected_classes:
                    default_value = suggestion.get(cls, 0)
                    desired = st.number_input(f"Para la clase '{cls}', número deseado:", min_value=0, value=int(default_value))
                    desired_samples[cls] = desired

            if mode_bbox:
                st.info("Modo segmentación/bounding box activado.")
                if objects_source == "Carpeta de objetos":
                    st.write("Utilizando objetos de la carpeta externa:")
                    st.write(default_objects_path)
                else:
                    st.write("Utilizando objetos extraídos del dataset de entrada.")
                st.write("Fondo utilizado:")
                st.write(default_images_path)
            else:
                st.info("Modo tradicional: se utilizarán las imágenes originales del dataset.")
            
            st.write("Ruta de salida para el dataset sintético:")
            st.write(default_output_dir)
            
            if st.button("Ejecutar Augmentación"):
                if not os.path.exists(default_output_dir):
                    os.makedirs(default_output_dir)
                
                progress_bar = st.progress(0)
                augmentor = SyntheticDataAugmentor(
                    output_dir=default_output_dir,
                    rot=rot,
                    scale=scale,
                    trans=trans,
                    try_count=try_count,
                    overlap_threshold=overlap_threshold
                )
                
                # Se pasa desired_synthetic_per_class como un diccionario con las muestras deseadas para cada clase.
                with st.spinner("Ejecutando proceso de augmentación..."):
                    if mode_bbox:
                        if objects_source == "Dataset de entrada":
                            synthetic_counts, synthetic_total = augmentor.augment_dataset(
                                coco_data, 
                                images_path=default_images_path, 
                                selected_classes=selected_classes,
                                objects_source="dataset",
                                backgrounds_dataset_path=default_images_path,
                                max_objects_per_image=max_obj_per_image,
                                desired_synthetic_per_class=desired_samples,
                                progress_bar=progress_bar
                            )
                        else:
                            synthetic_counts, synthetic_total = augmentor.augment_dataset(
                                coco_data, 
                                images_path=default_images_path, 
                                selected_classes=selected_classes,
                                objects_source="folder",
                                objects_dataset_path=default_objects_path,
                                backgrounds_dataset_path=default_images_path,
                                max_objects_per_image=max_obj_per_image,
                                desired_synthetic_per_class=desired_samples,
                                progress_bar=progress_bar
                            )
                    else:
                        synthetic_counts, synthetic_total = augmentor.augment_dataset(
                            coco_data, 
                            images_path=default_images_path, 
                            selected_classes=selected_classes,
                            desired_synthetic_per_class=desired_samples,
                            progress_bar=progress_bar
                        )
                
                st.success("Proceso de augmentación completado.")
                
                fig_synth = plot_synthetic_counts(synthetic_counts)
                st.subheader("Muestras sintéticas generadas por clase")
                st.pyplot(fig_synth)
                
                final_counts = {}
                for cls in set(original_counts.keys()).union(synthetic_counts.keys()):
                    final_counts[cls] = original_counts.get(cls, 0) + synthetic_counts.get(cls, 0)
                fig_final = plot_final_composition(original_counts, synthetic_counts)
                st.subheader("Composición final del dataset (original + sintéticas)")
                st.pyplot(fig_final)
                
                st.info("Revisa el directorio de salida para ver las nuevas imágenes, anotaciones y las imágenes anotadas.")
                
        except Exception as e:
            st.error(f"Error al procesar el archivo COCO: {e}")

if __name__ == "__main__":
    main()
