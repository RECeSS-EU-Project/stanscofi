---
title: 'A new standard for drug repurposing by collaborative filtering: stanscofi and benchscofi'
tags:
  - Python
  - drug repurposing
  - collaborative filtering
  - open science
  - science reproducibility
authors:
  - name: Clémence Réda
    orcid: 0000-0003-3238-0258
    affiliation: 1 
  - name: Jill-Jênn Vie
    orcid: 0000-0002-9304-2220
    affiliation: 2 
  - name: Olaf Wolkenhauer
    orcid: 0000-0001-6105-2937
    affiliation: "1,3,4" 
affiliations:
 - name: Department of Systems Biology and Bioinformatics, University of Rostock, Rostock, G-18055, Germany
   index: 1
 - name: Soda Team, Inria Saclay, F-91120 Palaiseau, France
   index: 2
 - name: Leibniz-Institute for Food Systems Biology, Freising, G-85354, Germany
   index: 3
 - name: Stellenbosch Institute of Advanced Study, Wallenberg Research Centre, Univ. Stellenbosch, Stellenbosch, SA-7602, South Africa
   index: 4
date: 5 September 2023
bibliography: paper.bib
---

# Statement of need

As of 2023, current drug development pipelines last around 10 years, costing $2.3billion in average [@philippidis2023unbearable], while drug commercialization failure rates go up to 90% [@sun202290]. These issues can be mitigated by drug repurposing, where chemical compounds are screened for new therapeutic indications in a systematic fashion. Indeed, documentation about those compounds is available, and helps prevent negative side effects and low accrual in clinical trials. These are still the main reasons for failure in late clinical phases [@hingorani2019improving]. In prior works [@jarada2020review;@zhang2017computational;@meng2022weighted;@yang2019additional], this approach has been implemented through collaborative filtering. This semi-supervised learning framework leverages known drug-disease matchings in order to recommend new ones. Predicted drug-disease associations stem from a function whose parameters are learnt on a whole matrix of drug-disease matchings, instead of focusing on a single disease at a time. In particular, literature often relies on tensor decomposition, *i*.*e*., any drug-disease matching in the matrix is the output of a classifier in which only lower-rank tensors intervene, *e*.*g*., factorization machines (FMs) [@rendle2010factorization]. 

Recent works [@he2020hybrid;@yang2022computational;@yang2023self] have reported near-perfect predicting power (*area under the curve*, or AUC) on several repurposing datasets. However, a considerable hurdle to the development of efficient drug repurposing approaches based on collaborative filtering is the lack of a standard pipeline to train, validate and compare this type of algorithms on a robust set of data.

The **stanscofi** Python package [@reda2023stanscofi] comprises method-agnostic training and validation procedures on several public drug repurposing datasets. The proper implementation of these steps is crucial in order to avoid data leakage, *i*.*e*., the model is learnt over information that should be unavailable at prediction time. Indeed, data leakage is the source of a major reproducibility crisis in machine learning [@roelofs2019meta;@feldman2019advantages;@kapoor2023leakage]. This will be avoided by two means: first, by building training and validation sets which are weakly correlated with respect to the drug and disease feature vectors; second, by implementing a generic model class which allows to automate as much as possible the training and validation procedures.

We also propose Python package **benchscofi** which builds upon the former package by wrapping the original implementations of 18 drug repurposing algorithms from the state of the art. To the best of our knowledge, this is the first time that such a package enables a large-scale benchmark of collaborative filtering-based drug repurposing approaches.

# Summary

The modularity of **stanscofi** and **benchscofi** at model, dataset and preprocessing level allows to straightforward enrich the package with newer and more efficient approaches. Those packages only consider and write files with the standard and versatile file format *csv*, which enables easy pipelining. Moreover, they allow to get access to several public drug repurposing datasets (see Table 1), and state of the art drug repurposing algorithms (a subset of which are displayed in Table 2) from various platforms in a straightforward way. **stanscofi** is built around four main modules, which are described in the remainder of the section.

## Module *datasets*

To facilitate benchmarking, stanscofi allows to easily import several drug repurposing datasets (see Table 1), all under the same form: a drug-disease matrix that summarizes reported clinical trials as either "positive" (denoted by a 1, for drugs which are known to treat the corresponding disease), "negative" (denoted by a -1, for clinical trials where toxic side effects or low accrual, for instance, were reported), and "unknown" (denoted by a 0, the most occurring outcome). Some of those datasets also comprise drug and disease features matrices, which brings supplementary information about drug-to-drug and disease-to-disease similarities. The datasets presented in **Table 1** are Gottlieb [@gottlieb2011predict] --also called FDataset in [@luo2018computational]-- LRSSL [@liang2017lrssl], CDataset, DNDataset [@luo2018computational], PREDICT-Gottlieb [@gao2022dda] --which is a version of FDataset with novel types of drug and disease features-- PREDICT [@reda2023PREDICT], and TRANSCRIPT [@reda2023TRANSCRIPT]. Moreover, an drug repurposing dataset can easily be converted into the *Dataset* class in **stanscofi**. This package also integrates several plotting functions which allows easier visualization of the datasets.

**Table 1:** Datasets in **stanscofi**. Reported numbers of drugs and diseases correspond to the number of drugs and diseases involved in at least one nonzero drug-disease matching. The sparsity number is the percentage of known (positive and negative) matchings times 100 over the total number of possible drug-disease matchings (rounded up to the second decimal place). 

Dataset            |   drugs       | diseases      | positive  | negative  | sparsity
-------------------|---------------|---------------|-----------|-----------|----------
PREDICT            | 1,351         | 1,066         |   5,624   |  152      | 0.34%
(nb. features)     |  (6,265)      |       (2,914) |           |           | 
TRANSCRIPT         | 204           | 116           |   401     |  11       | 0.45%
(nb. features)     |  (12,096)     |   (12,096)    |           |           | 
Gottlieb           | 593           | 313           |  1,933    |    0      | 1.04%
(nb. features)     |  (593)        | (313)         |           |           | 
CDataset           | 663           | 409           |  2,532    |     0     | 0.93%
(nb. features)     |  (663)        | (409)         |           |           | 
DNDataset          | 550           | 360           | 1,008     |     0     | 0.01%
(nb. features)     |  (1,490)      | (4,516)       |           |           | 
LRSSL              | 763           | 681           | 3,051     |      0    | 0.59%
(nb. features)     |  (2,049)      | (681)         |           |           | 
PREDICT-Gottlieb   | 593           | 313 (313)     | 1,933     |    0      | 1.04%
(nb. features)     |  (1,779)      | (313)         |           |           | 

## Module *training/testing*

As previously mentioned in introduction, **stanscofi** implements two approaches to splitting the dataset into training and validation sets. Along with the standard data splitting at random (function *random_simple_split*), it proposes for the first time the splitting into weakly correlated datasets (function *weakly_correlated_split*). This function is based on the hierarchical clustering of drugs based on their features, and the application of a bisection procedure in order to determine which cut in the dendrogram ensures that the size of the validation set is almost equal to the user-specified value (for instance, 20\% of known drug-disease matchings).

**stanscofi** also provides readily usable functions for cross-validation (function *cv_training*) and grid searches for hyperparameters (*grid_search*). 

## Module *models*

**stanscofi** implements a **BasicModel** class which takes as input **stanscofi** *Dataset* objects, and permits to fit (class method *fit*), to score (*predict_proba*), to label (*predict*) in a fashion which is similar to well-known Python machine learning packages such as **scikit-learn** [@pedregosa2011scikit]. However, contrary to **scikit-learn** procedures, these functions can also handle non-binary outcomes, as often the case in collaborative filtering (with values -1, 0, and 1). Furthermore, the **BasicModel** class can also tackle recommendation-specific tasks (*e*.*g*., to recommend the top *k* drug-disease pairs with method *recommend_k_pairs*).

## Module *validation*

**stanscofi** evaluates metrics on a testing dataset through function *compute_metrics*, which can be combined with function *plot_metrics* in order to visualize at a glance the disease-wise Receiver Operating Characteristic (ROC) and Precision-Recall curves, a boxplot of scores obtained on the testing dataset, and the accuracy of predictions on known ratings. The main considered performance metric is the standard *area under the curve* (AUC) metric, along with an average disease-wise version. This disease-wise version first estimates the diagnostic ability of a recommender system better than accuracy in imbalanced datasets [@ling2003auc]; and, second, takes into account the variation in predictive power across diseases. **stanscofi** also includes other standard accuracy and ranking metrics, such as F-score, mean rank, or normalized discounted cumulative gain (globally, or at a specific rank).

Using that package, one can then test algorithms from the literature and easily develop a benchmark pipeline, which is what we demonstrated by implementating package **benchscofi**. 

## *benchscofi* package

We have compiled 18 collaborative filtering algorithms from the literature in **benchscofi** [@reda2023benchscofi]. Those cover a large part of platforms (R, MATLAB, Python) and approaches (matrix factorization, graph-based methods, etc.). We report in Table 2 below some of the results obtained using this package for the following algorithms, using default parameter values: PMF [@mnih2007probabilistic], ALSWR [@ethen2023ALSWR], Fast.ai collab learner [@howard2020deep], NIMCGCN [@li2020neural], DRRS [@luo2018computational], SCPMF [@meng2021drug], BNNR [@yang2019drug], LRSSL [@liang2017lrssl], MBiRW [@luo2016drug], LibMF [@chin2016libmf], LogisticMF [@johnson2014logistic], DDA-SKF [@gao2022dda] and HAN [@wang2019heterogeneous]. The code to execute in order to get **Table 2** is ''cd tests/ && python3 -m test_models ALGORITHM DATASET'', where ALGORITHM and DATASET are replaced by the corresponding row and column indices.

**Table 2:** Results obtained by combining **stanscofi** and **benchscofi**. Reported values are the standard *area under the curve* (AUC) scores, which are globally computed on all scores associated with drug-disease pairs. Maximum value in a column is denoted with an asterisk.

  Algorithm  (AUC)         | TRANSCRIPT        | Gottlieb      | CDataset     | LRSSL      
-------------------------- | ----------------- | ------------- | ------------ | ----------
ALSWR                      |  0.507            |  0.677        |  0.724       |  0.685   
BNNR                       |  0.922 *          |  0.949        |  0.959  *    |  0.972     
DDA-SKF                    |  0.453            |  0.544        |  0.264       |  0.542     
DRRS                       |  0.662            |  0.838        |  0.878       |  0.892  
Fast.ai collab learner     |  0.876            |  0.856        |  0.837       |  0.851     
HAN                        |  0.870            |  0.909        |  0.905       |  0.923   
LibMF                      |  0.919            |  0.892        |  0.912       |  0.873  
LogisticMF                 |  0.910            |  0.941        |  0.955       |  0.933     
LRSSL                      |  0.581            |  0.159        |  0.846       |  0.665   
MBiRW                      |  0.913            |  0.954  *     |  0.965       |  0.975  *   
NIMCGCN                    |  0.854            |  0.843        |  0.841       |  0.873      
PMF                        |  0.579            |  0.598        |  0.604       |  0.611    
SCPMF                      |  0.680            |  0.548        |  0.538       |  0.708     

This package allows the design of large-scale benchmarks, and enables a fair and comprehensive assessment of the performance of state of the art methods. It will ease the development and testing of competitive drug repurposing approaches. A collection of Jupyter notebooks aiming at uncovering the different components implemented in **stanscofi** and **benchscofi** is available on the corresponding GitHub repositories.

The two packages **stanscofi** and **benchscofi** have the potential to significantly alleviate the economic burden of drug discovery pipelines. They could help to find treatments in a more sustainable manner, which still remains a topical question, especially for rare or tropical neglected diseases [@walker2021supporting]. This project is in line with recent European health policies, in which drug repurposing has become a top priority in 2020 [@europeanpolicy].

# Acknowledgements

The research leading to these results has received funding from the European Union’s HORIZON 2020 Programme under grant agreement no. 101102016 (RECeSS, HORIZON TMA MSCA Postdoctoral Fellowships - European Fellowships, C.R.).

# References
