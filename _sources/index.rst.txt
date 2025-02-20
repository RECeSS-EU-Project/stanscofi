.. stanscofi documentation master file, created by
   sphinx-quickstart on Fri Jul 14 14:47:41 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the documentation of packages *stanscofi* and *benchscofi*
======================================================================

.. image:: images/header+EU_rescale.jpg
  :width: 700
  :alt: RECeSS project

These packages are a part of the EU-funded **Robust Explainable Controllable Standard for drug Screening** (**RECeSS**) project `(RECeSS#101102016) <https://recess-eu-project.github.io>`_, and hosts the code for the open-source Python packages *stanscofi* for the development of collaborative filtering-based drug repurposing algorithms and *benchscofi* for the implementations of state-of-the-art algorithms.

As of 2022, current drug development pipelines last around 10 years, costing $2billion in average, while drug commercialization failure rates go up to 90%. These issues can be mitigated by drug repurposing, where chemical compounds are screened for new therapeutic indications in a systematic fashion. In prior works, this approach has been implemented through collaborative filtering. This semi-supervised learning framework leverages known drug-disease matchings in order to recommend new ones. Medium-term outcomes to these two packages will significantly alleviate the economic burden of drug discovery pipelines, and will help find treatments in a more sustainable manner, especially for rare or tropical neglected diseases. For more information about the datasets accessible in *stanscofi*, please refer to the following `repository <https://github.com/RECeSS-EU-Project/drug-repurposing-datasets>`__.

Licence & Citation
-------------------

Those two packages are under an OSI-approved MIT license. If you use them in academic research, please cite them as follows: ::

   Réda, Clémence, Jill-Jênn Vie, and Olaf Wolkenhauer. (2024).
   "stanscofi and benchscofi: a new standard for drug repurposing by collaborative filtering." 
   Journal of Open Source Software, 9(93), 5973, https://doi.org/10.21105/joss.05973
  
Package *stanscofi*
=====================

The *stanscofi* package comprises method-agnostic training, validation, preprocessing and visualization procedures on several published drug repurposing datasets. The proper implementation of these steps is crucial in order to avoid data leakage, i.e., the model is learnt over information that should be unavailable at prediction time. Indeed, data leakage is the source of a major reproducibility crisis in machine learning. This will be avoided by building training and validation sets which are weakly correlated with respect to the drug and disease feature vectors. The main performance metric will be the area under the curve (AUC), which estimates the diagnostic ability of a recommender system better than accuracy in imbalanced datasets.

.. toctree::
   :maxdepth: 2
   
   stanscofi_install
   stanscofi_content
   stanscofi_modules

Package *benchscofi*
=====================

Moreover, there is no standard pipeline to train, validate and compare collaborative filtering-based repurposing methods, which considerably limits the impact of this research field. In the package *benchscofi*, which is built upon *stanscofi*, the estimated improvement over the state-of-the-art (as implemented in the package) can be measured through adequate and quantitative metrics tailored to the problem of drug repurposing across a large set of publicly available drug repurposing datasets.

.. toctree::
   :maxdepth: 2
   
   benchscofi_install
   benchscofi_content

External Links
==================

* :ref:`genindex`
* `GitHub Project Repository for stanscofi <https://github.com/recess-eu-project/stanscofi>`_
* `GitHub Project Repository for benchscofi <https://github.com/recess-eu-project/benchscofi>`_
* `RECeSS Project Website <https://recess-eu-project.github.io>`_
