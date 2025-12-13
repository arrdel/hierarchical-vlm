"""Setup configuration for HierarchicalVLM."""

from setuptools import setup, find_packages

setup(
    name="hierarchicalvlm",
    version="1.0.0",
    description="Hierarchical Vision-Language Model for long-context video understanding",
    author="HierarchicalVLM Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.13.0",
        "torchvision>=0.14.0",
        "transformers>=4.25.0",
        "numpy>=1.20.0",
        "tensorboard>=2.10.0",
        "peft>=0.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
        ],
    },
)
