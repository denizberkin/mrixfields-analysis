from setuptools import setup, find_packages

setup(
    name="mrixfields",
    version="0.1.0",
    description="MRIxFields2026: A Generalizable Cross-Field MRI Translation and Harmonization Challenge",
    author="MRIxFields2026 Organizing Team",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0,<2.5",
        "torchvision>=0.15.0,<0.20",
        "tensorflow>=2.15.0,<2.16",
        "keras>=2.15.0,<2.16",
        "nibabel>=5.0.0",
        "numpy>=1.24.0,<2.0",
        "scipy>=1.10.0",
        "scikit-image>=0.20.0",
        "matplotlib>=3.7.0",
        "lpips>=0.1.4",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "SimpleITK>=2.2.0",
        "six>=1.16.0",
        "h5py>=3.0.0",
    ],
)
