![funding logo](https://raw.githubusercontent.com/RECeSS-EU-Project/RECeSS-EU-Project.github.io/main/assets/images/header%2BEU_rescale.jpg)

# STANdard for drug Screening by COllaborative FIltering (stanscofi) Python Package

This repository is a part of the EU-funded [RECeSS project](https://recess-eu-project.github.io) (#101102016), and hosts the code for the open-source Python package *stanscofi* for the development of collaborative filtering-based drug repurposing algorithms.

[![Python Version](https://img.shields.io/badge/python-3.8%7C3.9-pink)](https://badge.fury.io/py/stanscofi) [![PyPI version](https://img.shields.io/pypi/v/stanscofi.svg)](https://badge.fury.io/py/stanscofi) [![Anaconda version](https://anaconda.org/recess/stanscofi/badges/version.svg)](https://anaconda.org/recess/stanscofi) [![Zenodo version](https://zenodo.org/badge/DOI/10.5281/zenodo.8038847.svg)](https://doi.org/10.5281/zenodo.8038847) ![GitHub](https://img.shields.io/github/license/recess-eu-project/stanscofi.svg) [![Build Status](https://github.com/recess-eu-project/stanscofi/actions/workflows/post-push-test.yml/badge.svg)](https://github.com/recess-eu-project/stanscofi/actions/workflows/post-push-test.yml) [![Codecov](https://codecov.io/github/recess-eu-project/stanscofi/coverage.svg?branch=master)](https://codecov.io/github/recess-eu-project/stanscofi?branch=master) [![Codefactor](https://www.codefactor.io/repository/github/recess-eu-project/stanscofi/badge?style=plastic)](https://www.codefactor.io/repository/github/recess-eu-project/stanscofi)

## Statement of need 

As of 2022, current drug development pipelines last around 10 years, costing $2billion in average, while drug commercialization failure rates go up to 90%. These issues can be mitigated by drug repurposing, where chemical compounds are screened for new therapeutic indications in a systematic fashion. In prior works, this approach has been implemented through collaborative filtering. This semi-supervised learning framework leverages known drug-disease matchings in order to recommend new ones.

The **stanscofi** package comprises method-agnostic training, validation, preprocessing and visualization procedures on several published drug repurposing datasets. The proper implementation of these steps is crucial in order to avoid data leakage, *i*.*e*., the model is learnt over information that should be unavailable at prediction time. Indeed, data leakage is the source of a major reproducibility crisis in machine learning. This will be avoided by building training and validation sets which are weakly correlated with respect to the drug and disease feature vectors. The main performance metric will be the area under the curve (AUC), which estimates the diagnostic ability of a recommender system better than accuracy in imbalanced datasets.

Medium-term outcomes to this package will significantly alleviate the economic burden of drug discovery pipelines, and will help find treatments in a more sustainable manner, especially for rare or tropical neglected diseases.

For more information about the datasets accessible in **stanscofi**, please refer to the following [repository](https://github.com/RECeSS-EU-Project/drug-repurposing-datasets).

## Install the latest release

### Using pip (package hosted on PyPI)

```bash
pip install stanscofi
```

### Using conda (package hosted on Anaconda.org)

```bash
conda install -c recess stanscofi
```

## Running the notebooks

## Example usage

Once installed, to import **stanscofi** into your Python code

```
import stanscofi
```

Please check out notebook *Introduction to stanscofi.ipynb*. Documentation about **stanscofi** can be found at [this page](https://recess-eu-project.github.io/stanscofi/). All functions are documented, so one can check out the inputs and outputs of a function func by typing

```
help(func)
```

To mesure your environmental impact when using this package (in terms of carbon emissions), please run the following command

```
! codecarbon init
```

 to initialize the CodeCarbon config. For more information about using CodeCarbon, please refer to the [official repository](https://github.com/mlco2/codecarbon).

### Environment

In order to run notebook *Introduction to stanscofi.ipynb*, it is strongly advised to create a virtual environment using Conda (python>=3.8)

```
conda create -n stanscofi_env python=3.8.5 -y
conda activate stanscofi_env
python3 -m pip install stanscofi ## or use the conda command above
python3 -m pip install notebook>=6.5.4 markupsafe==2.0.1 ## packages for Jupyter notebook
conda deactivate
conda activate stanscofi_env
cd docs/ && jupyter notebook
```

The complete list of dependencies for *stanscofi* can be found at [requirements.txt](https://raw.githubusercontent.com/RECeSS-EU-Project/stanscofi/master/pip/requirements.txt) (pip) or [meta.yaml](https://raw.githubusercontent.com/RECeSS-EU-Project/stanscofi/master/conda/meta.yaml) (conda).

## Licence

This repository is under an [OSI-approved](https://opensource.org/licenses/) [MIT license](https://raw.githubusercontent.com/RECeSS-EU-Project/stanscofi/master/LICENSE). 

## Citation

If you use **stanscofi** in academic research, please cite it as follows

```
Réda, Clémence. (2023). 
STANdard for drug Screening by COllaborative FIltering (stanscofi). 
Zenodo. https://doi.org/10.5281/zenodo.8038847
```

## Community guidelines with respect to contributions, issue reporting, and support

[Pull requests](https://github.com/RECeSS-EU-Project/stanscofi/pulls) and [issue flagging](https://github.com/RECeSS-EU-Project/stanscofi/issues) are welcome, and can be made through the GitHub interface. Support can be provided by reaching out to ``recess-project[at]proton.me``. However, please note that contributors and users must abide by the [Code of Conduct](https://github.com/RECeSS-EU-Project/stanscofi/blob/master/CODE%20OF%20CONDUCT.md).

