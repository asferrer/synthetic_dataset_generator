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

# Define default paths from config
default_images_path = config["paths"]["backgrounds_dataset"]  # Background images folder
default_output_dir = os.path.dirname(config["paths"]["images"])  # Synthetic output folder
default_objects_path = config["paths"].get("objects_dataset", None)  # External objects folder

def plot_synthetic_counts(synthetic_counts):
    classes = list(synthetic_counts.keys())
    counts = list(synthetic_counts.values())
    fig, ax = plt.subplots()
    ax.bar(classes, counts, color='orange')
    ax.set_title("Synthetic Samples Generated per Class")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Synthetic Instances")
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
    ax.set_title("Final Dataset Composition (Original + Synthetic)")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Instances")
    plt.xticks(rotation=45, ha='right')
    return fig

def main():
    st.title("Synthetic Data Augmentation Tool")
    
    st.sidebar.header("Augmentation Configuration")
    rotate_flag = st.sidebar.checkbox("Rotation", value=config["augmentation"]["rot"])
    scale_flag = st.sidebar.checkbox("Scaling", value=config["augmentation"]["scale"])
    translate_flag = st.sidebar.checkbox("Translation", value=config["augmentation"]["trans"])
    try_count = st.sidebar.number_input("Maximum Paste Attempts", min_value=1, max_value=10, value=config["augmentation"]["try_count"])
    overlap_threshold = st.sidebar.number_input("Overlap Threshold (%)", min_value=0, max_value=100, value=config["augmentation"]["overlap_threshold"])
    minority_threshold = st.sidebar.number_input("Minority Class Threshold (number of instances)", min_value=1, value=1500)
    
    max_obj_per_image = st.sidebar.number_input("Max Objects per Image", min_value=1, value=config["augmentation"].get("max_objects_per_image", 3))
    
    st.sidebar.subheader("Object Source for Augmentation")
    objects_source = st.sidebar.radio("Select object source:", options=["External Folder","Input Dataset"])
        
    save_intermediate = st.sidebar.checkbox("Save Intermediate Steps", value=False)

    st.header("COCO Dataset Upload and Analysis")
    st.markdown("Upload your COCO JSON annotation file.")
    
    coco_file = st.file_uploader("Select your COCO JSON file", type=["json"])
    
    if coco_file is not None:
        try:
            coco_data = json.load(coco_file)
            st.success("COCO file loaded successfully.")
            
            analysis = analyze_coco_dataset(coco_data)
            st.subheader("Original Dataset Summary")
            st.write(f"Number of images: {analysis.get('num_images', 'N/A')}")
            st.write(f"Number of annotations: {analysis.get('num_annotations', 'N/A')}")
            original_counts = analysis.get("class_counts", {})
            st.write("Original Class Distribution:")
            st.write(original_counts)
            
            fig_orig = plot_class_distribution(original_counts)
            st.pyplot(fig_orig)
            
            # Suggest synthetic sample counts per class based on input dataset
            if original_counts:
                target_count = max(original_counts.values())
                suggestion = {cls: max(target_count - original_counts.get(cls, 0), 0) for cls in original_counts.keys()}
                st.subheader("Suggested Synthetic Samples per Class")
                st.write("It is suggested to generate these synthetic samples to balance the dataset:")
                st.write(suggestion)
            else:
                suggestion = {}

            minority_classes = [cls for cls, count in original_counts.items() if count < minority_threshold]
            st.write(f"Minority Classes (less than {minority_threshold} instances):")
            st.write(minority_classes)
            
            selected_classes = st.multiselect("Select classes to augment", minority_classes)
            
            # For each selected class, allow the user to define the desired number of synthetic samples.
            desired_samples = {}
            if selected_classes:
                st.subheader("Define Desired Synthetic Samples per Class")
                for cls in selected_classes:
                    default_value = suggestion.get(cls, 0)
                    desired = st.number_input(f"For class '{cls}', desired number:", min_value=0, value=int(default_value))
                    desired_samples[cls] = desired
            
            st.write("Synthetic dataset output directory:")
            st.write(default_output_dir)
            
            if st.button("Run Augmentation"):
                if not os.path.exists(default_output_dir):
                    os.makedirs(default_output_dir)
                
                progress_bar = st.progress(0)
                augmentor = SyntheticDataAugmentor(
                    output_dir=default_output_dir,
                    rot=rotate_flag,
                    scale=scale_flag,
                    trans=translate_flag,
                    try_count=try_count,
                    overlap_threshold=overlap_threshold/100,
                    save_intermediate_steps=save_intermediate
                )
                
                with st.spinner("Running augmentation process..."):
                    if objects_source == "Input Dataset":
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
                
                st.success("Augmentation process completed.")
                
                fig_synth = plot_synthetic_counts(synthetic_counts)
                st.subheader("Synthetic Samples Generated per Class")
                st.pyplot(fig_synth)
                
                final_counts = {}
                for cls in set(original_counts.keys()).union(synthetic_counts.keys()):
                    final_counts[cls] = original_counts.get(cls, 0) + synthetic_counts.get(cls, 0)
                fig_final = plot_final_composition(original_counts, synthetic_counts)
                st.subheader("Final Dataset Composition (Original + Synthetic)")
                st.pyplot(fig_final)
                
                st.info("Check the output directory for new images, annotations, and annotated images.")
                
        except Exception as e:
            st.error(f"Error processing the COCO file: {e}")

if __name__ == "__main__":
    main()
