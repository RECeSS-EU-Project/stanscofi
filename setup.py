from setuptools import setup, find_packages

NAME = "stanscofi"
VERSION = "9999"

setup(name=NAME,
    version=VERSION,
    author="Clémence Réda",
    author_email="recess-project@proton.me",
    url="https://github.com/RECeSS-EU-Project/stanscofi",
    license_files = ('LICENSE'),
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
    ],
    keywords='',
    description="Package for STANdard drug Screening by COllaborative FIltering. Performs benchmarks against datasets and SotA algorithms, and implements training, validation and testing procedures.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={'':"src"},
    python_requires='>=3.8.5',
    install_requires=[
        "cute-ranking>=0.0.3",
        "scikit-learn>=1.3.0",
        "matplotlib",
        "tqdm>=4.65.0",
        "codecarbon>=2.2.5",
        "umap-learn>=0.5.3",
        "seaborn>=0.12.2",
    ],
    entry_points={},
)
