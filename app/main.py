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
    """Carga la configuración desde un archivo YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = load_config()

# Rutas por defecto desde config
default_backgrounds_path = config["paths"]["backgrounds_dataset"]
default_output_dir = os.path.dirname(config["paths"]["images"])
default_objects_path = config["paths"].get("objects_dataset", "")

def plot_bar_chart(data, title, xlabel, ylabel, color):
    if not data:
        return None
    classes = list(data.keys())
    counts = list(data.values())
    fig, ax = plt.subplots()
    ax.bar(classes, counts, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


def get_available_classes_from_folder(path):
    if not path or not os.path.isdir(path):
        return []
    return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])


def parse_class_mapping(mapping_text):
    mapping = {}
    if not mapping_text:
        return mapping
    lines = mapping_text.strip().split("\n")
    for line in lines:
        if ":" in line:
            target, sources = line.split(":", 1)
            tgt = target.strip()
            srcs = [s.strip() for s in sources.split(",") if s.strip()]
            if tgt and srcs:
                mapping[tgt] = srcs
    return mapping


def run_augmentation_process(augmentor, augment_params):
    with st.spinner("Ejecutando proceso de aumentación... Esto puede tardar varios minutos."):
        synthetic_counts, synthetic_total = augmentor.augment_dataset(**augment_params)

    st.success("✅ Proceso de aumentación completado.")
    st.balloons()
    st.subheader("Resultados de la Generación")

    original_counts = augment_params.get("original_counts", {})
    col1, col2 = st.columns(2)
    with col1:
        st.write("Muestras Sintéticas Generadas por Clase:")
        fig_synth = plot_bar_chart(
            synthetic_counts,
            "Muestras Sintéticas Generadas",
            "Clase",
            "Nº de Instancias Sintéticas",
            "orange"
        )
        if fig_synth:
            st.pyplot(fig_synth)
    with col2:
        st.write("Composición Final del Dataset (Original + Sintético):")
        final_counts = {cls: original_counts.get(cls, 0) + synthetic_counts.get(cls, 0)
                        for cls in set(original_counts) | set(synthetic_counts)}
        fig_final = plot_bar_chart(
            final_counts,
            "Composición Final del Dataset",
            "Clase",
            "Instancias Totales",
            "mediumseagreen"
        )
        if fig_final:
            st.pyplot(fig_final)

    st.info(f"Revisa el directorio de salida `{default_output_dir}` para ver las nuevas imágenes, anotaciones y el JSON global.")


def main():
    st.set_page_config(layout="wide")
    st.title("Herramienta de Aumentación de Datos Sintéticos con Realismo Mejorado")

    # --- Configuración de la Barra Lateral ---
    st.sidebar.header("Configuración de Aumentación")
    with st.sidebar.expander("Transformaciones Básicas"):
        rotate_flag = st.checkbox("Rotación", value=config["augmentation"]["rot"])
        scale_flag = st.checkbox("Escalado", value=config["augmentation"]["scale"])
        translate_flag = st.checkbox("Translación", value=config["augmentation"]["trans"])
    with st.sidebar.expander("Parámetros de Generación"):
        try_count = st.number_input(
            "Máximos Intentos de Pegado", min_value=1, max_value=20,
            value=config["augmentation"]["try_count"]
        )
        overlap_threshold = st.number_input(
            "Umbral de Solapamiento (%)", min_value=0, max_value=100,
            value=config["augmentation"]["overlap_threshold"]
        )
        max_obj_per_image = st.number_input(
            "Máx Objetos por Imagen", min_value=1,
            value=config["augmentation"].get("max_objects_per_image", 5)
        )
        max_upscale_ratio = st.number_input(
            "Ratio Máximo de Escalado", 1.0, 10.0,
            3.0, 0.5
        )
    st.sidebar.header("Mejoras de Realismo y Calidad")
    with st.sidebar.expander("Opciones de Realismo", expanded=True):
        realism_intensity = st.slider(
            "Intensidad de Efectos de Realismo", 0.0, 1.0, 0.4, 0.05
        )
        underwater_effect = st.checkbox("Efecto Submarino (Velo Acuático)", value=False)
        advanced_color_correction = st.checkbox("Corrección de Color Suave", value=True)
        blur_consistency = st.checkbox("Consistencia de Desenfoque", value=False)
        add_shadows = st.checkbox("Generar Sombras", value=True)
        perspective_transform = st.checkbox("Transformación de Perspectiva", value=True)
        poisson_blending = st.checkbox("Poisson Blending (Opcional)", value=False)
        lighting_effects = st.checkbox("Efectos de Iluminación", value=False)
        motion_blur = st.checkbox("Desenfoque de Movimiento", value=False)
    
    st.sidebar.subheader("Control de Calidad de Objetos")
    min_area_ratio = st.sidebar.slider(
        "Ratio de Área Mínimo del Objeto", 
        min_value=0.0, max_value=0.1, value=0.005, step=0.001, format="%.3f",
        help="El tamaño mínimo que debe tener un objeto como porcentaje del área total de la imagen."
    )
    max_area_ratio = st.sidebar.slider(
        "Ratio de Área Máximo del Objeto", 
        min_value=0.1, max_value=1.0, value=0.4, step=0.05,
        help="El tamaño máximo que puede tener un objeto como porcentaje del área total de la imagen."
    )
    
    st.sidebar.header("Opciones de Salida")
    save_intermediate = st.sidebar.checkbox("Guardar Pasos Intermedios", value=False)

    # Modo de Operación
    mode = st.sidebar.radio(
        "Elige qué quieres hacer:",
        ("Balancear un dataset existente", "Generar un dataset desde cero"),
        key="operation_mode"
    )

    # Instanciar el aumentador
    augmentor = SyntheticDataAugmentor(
        output_dir=default_output_dir,
        rot=rotate_flag, scale=scale_flag, trans=translate_flag,
        poisson_blending=poisson_blending,
        advanced_color_correction=advanced_color_correction,
        blur_consistency=blur_consistency, add_shadows=add_shadows,
        perspective_transform=perspective_transform,
        lighting_effects=lighting_effects, motion_blur=motion_blur,
        underwater_effect=underwater_effect,
        try_count=try_count, overlap_threshold=overlap_threshold/100.0,
        save_intermediate_steps=save_intermediate,
        realism_intensity=realism_intensity,
        max_upscale_ratio=max_upscale_ratio,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio
    )

    if mode == "Balancear un dataset existente":
        st.header("Modo de Balanceo: Carga y Análisis del Dataset COCO")
        coco_file = st.file_uploader("Selecciona tu fichero COCO JSON", type=["json"])
        if coco_file:
            coco_data = json.load(coco_file)
            st.success("Fichero COCO cargado correctamente.")

            original_counts = analyze_coco_dataset(coco_data, {}).get("class_counts", {})
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Número de clases distintas", len(original_counts))
                st.write("Distribución de Clases Original:")
                st.dataframe(original_counts)
            with col2:
                fig_orig = plot_bar_chart(
                    original_counts, "Distribución Original", "Clase", "Instancias", "skyblue"
                )
                if fig_orig:
                    st.pyplot(fig_orig)

            st.subheader("Agrupación de Clases (Opcional)")
            st.markdown("Si las clases del JSON no coinciden con tu carpeta de objetos, define agrupaciones aquí.")
            available_objs = get_available_classes_from_folder(default_objects_path)
            st.info(f"Clases disponibles en carpeta de objetos: {', '.join(available_objs)}")
            default_map = "\n".join(
                f"{cls}: {cls}" for cls in original_counts.keys() if cls in available_objs
            )
            mapping_text = st.text_area(
                "Define agrupaciones (formato Target: src1, src2):",
                value=default_map,
                help="Ejemplo: Fish: Fish, Pike"
            )
            class_mapping = parse_class_mapping(mapping_text)

            grouped_counts = analyze_coco_dataset(coco_data, class_mapping).get("class_counts", {})
            st.write("Distribución tras agrupación:")
            st.dataframe(grouped_counts)

            st.subheader("Selección de Clases a Aumentar")
            minority_threshold = st.number_input("Umbral de Clase Minoritaria", min_value=1, value=1500)
            minority = [c for c, count in grouped_counts.items() if count < minority_threshold]
            st.info(f"Clases minoritarias (<{minority_threshold}): {', '.join(minority)}")
            selected = st.multiselect("Elige clases a aumentar", list(grouped_counts.keys()), default=minority)

            desired = {}
            if selected:
                st.subheader("Define nº de muestras sintéticas por clase")
                cols = st.columns(len(selected))
                for i, cls in enumerate(selected):
                    with cols[i]:
                        desired[cls] = st.number_input(
                            f"Para '{cls}'",
                            min_value=0,
                            value=max(0, max(grouped_counts.values()) - grouped_counts.get(cls, 0)),
                            key=f"bal_{cls}"
                        )

            objects_source = st.sidebar.radio("Fuente de los Objetos:", ["Input Dataset", "External Folder"])

            if st.button("🚀 Ejecutar Balanceo", use_container_width=True):
                if not selected:
                    st.error("Selecciona al menos una clase para aumentar.")
                else:
                    params = {
                        "coco_data": coco_data,
                        "images_path": default_backgrounds_path,
                        "selected_classes": selected,
                        "objects_source": objects_source.lower().replace(" ", "_"),
                        "objects_dataset_path": default_objects_path,
                        "backgrounds_dataset_path": default_backgrounds_path,
                        "max_objects_per_image": max_obj_per_image,
                        "desired_synthetic_per_class": desired,
                        "class_mapping": class_mapping,
                        "progress_bar": st.progress(0),
                        "status_text": st.empty(),
                        "original_counts": grouped_counts
                    }
                    run_augmentation_process(augmentor, params)
    else:
        st.header("Modo de Generación: Crear Dataset desde Cero")
        source_classes_in_folder = get_available_classes_from_folder(default_objects_path)
        if not source_classes_in_folder:
            st.warning("No se encontraron clases (subcarpetas) en la ruta de objetos especificada. Por favor, verifica la ruta.")
        else:
            with st.expander("Agrupación de Clases (Opcional)", expanded=True):
                st.markdown("Define cómo agrupar las clases de origen de tus carpetas en nuevas clases destino.")
                default_mapping = "\n".join(source_classes_in_folder)
                mapping_text = st.text_area(
                    "Define agrupaciones (una por línea):",
                    value=default_mapping,
                    help="Formato: CLASE_OBJETIVO: clase_origen_1, clase_origen_2, ..."
                )
                class_mapping = parse_class_mapping(mapping_text)

            mapped_source_classes = {src for srcs in class_mapping.values() for src in srcs}
            target_classes_from_mapping = list(class_mapping.keys())
            unmapped_classes = [cls for cls in source_classes_in_folder if cls not in mapped_source_classes]
            available_target_classes = sorted(set(target_classes_from_mapping + unmapped_classes))

            st.success(f"Clases de origen detectadas: {', '.join(source_classes_in_folder)}")
            st.info(f"Clases objetivo disponibles para generar: {', '.join(available_target_classes)}")

            selected_gen = st.multiselect("Selecciona las clases a generar", options=available_target_classes, default=available_target_classes)
            desired_gen = {}
            if selected_gen:
                st.subheader("Define el número de muestras sintéticas por clase")
                cols = st.columns(min(len(selected_gen), 4))
                for i, cls in enumerate(selected_gen):
                    with cols[i % 4]:
                        cnt = st.number_input(
                            f"Para '{cls}':",
                            min_value=0,
                            value=100,
                            key=f"desired_gen_{cls}"
                        )
                        if cnt > 0:
                            desired_gen[cls] = cnt

            st.write(f"**Directorio de salida del dataset sintético:** `{default_output_dir}`")

            if st.button("🚀 Generar Dataset desde Cero", use_container_width=True):
                if not desired_gen:
                    st.error("Por favor, especifica un número de muestras mayor que cero para al menos una clase.")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    augment_params = {
                        "coco_data": None,
                        "images_path": None,
                        "selected_classes": list(desired_gen.keys()),
                        "objects_source": "folder",
                        "objects_dataset_path": default_objects_path,
                        "backgrounds_dataset_path": default_backgrounds_path,
                        "max_objects_per_image": max_obj_per_image,
                        "desired_synthetic_per_class": desired_gen,
                        "progress_bar": progress_bar,
                        "status_text": status_text,
                        "class_mapping": class_mapping
                    }
                    run_augmentation_process(augmentor, augment_params)

if __name__ == "__main__":
    main()
