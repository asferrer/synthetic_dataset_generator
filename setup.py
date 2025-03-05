import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="synthetic_data_augmentation_tool",
    version="0.1.0",
    author="Alejandro Sanchez Ferrer",
    author_email="asanc.tech@gmail.com",
    description="Herramienta para aumentar datos sintéticos a partir de datasets etiquetados",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/asferrer/synthetic_dataset_generator",
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "opencv-python",
        "scipy",
        "imageio",
        "tqdm",
        "matplotlib",
        "streamlit",
    ],
    entry_points={
        "console_scripts": [
            "augment=app.main:main",  # Se asume que en app/main.py defines una función main()
        ],
    },
)