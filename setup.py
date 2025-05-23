from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="spikeflow",
    version="0.1.0",
    author="JonusNattapong",
    author_email="jonus@example.com",
    description="Enhanced Spiking Neural Network Library for Neuromorphic Computing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JonusNattapong/SpikeFlow",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.60.0",
        "tensorboard>=2.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "black>=21.0",
            "flake8>=3.8",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
        "hardware": [
            "nxsdk",  # Intel Loihi SDK
            "spynnaker",  # SpiNNaker interface
        ],
    },
)
