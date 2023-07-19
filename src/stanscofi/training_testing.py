#coding: utf-8

import pandas as pd
import random
import numpy as np
from copy import deepcopy
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.sparse import coo_array

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import StratifiedKFold, train_test_split

from .validation import compute_metrics, plot_metrics, metrics_list

#######################################
# Weak correlation / random splitting #
#######################################

def indices_to_folds(indices, indices_array, shape):
    '''
    Converts indices of datapoints into folds as defined in stanscofi

    ...

    Parameters
    ----------
    indices : array-like of size (n_selected_ratings, )
        flat indices of selected datapoints
    indices_array : array-like of size (n_total_ratings, 2)
        corresponding row and column indices of datapoints
    shape : tuple of integers of size 2
        total numbers of rows and columns

    Returns
    ----------
    folds : COO-array of shape shape
        folds which can be fed to other functions in stanscofi, e.g., dataset.subset(folds)
    '''
    row = indices_array[indices,0].ravel()
    col = indices_array[indices,1].ravel()
    data = np.ones(indices.shape[0])
    return coo_array((data, (row, col)), shape=shape)

def random_simple_split(dataset, test_size, metric="cosine", random_state=1234):
    '''
    Splits the data into training and testing datasets randomly.

    ...

    Parameters
    ----------
    dataset : stanscofi.Dataset
        dataset to split
    test_size : float
        value between 0 and 1 (strictly) which indicates the maximum percentage of initial data (positive and negative ratings) being assigned to the test dataset
    metric : str
        metric to consider to assess distance between training and testing sets. Should belong to [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’, ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘correlation’, ‘dice’, ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’]
    random_state : int
        random seed 

    Returns
    ----------
    cv_folds : list of COO-array of shape (n_items, n_users)
        list of arrays which contain values in {0, 1} describing the unavailable and available user-item matchings in the training (resp. testing) set
    dist_train_test, dist_train, dist_test : float
        minimum nonzero distance between an element in the training and in the testing sets, resp. inside the training set, resp. inside the testing set
    '''
    X = np.column_stack((dataset.folds.row, dataset.folds.col)) ## indices for available ratings
    y = dataset.ratings.toarray()[dataset.folds.row,dataset.folds.col].ravel() ## values for available ratings
    train_set, test_set,_,_ = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True, stratify=y)
    cv_folds = tuple([
                coo_array((np.ones(x.shape[0]), (x[:,0].ravel(), x[:,1].ravel())), 
                        shape=dataset.folds.shape) for x in [train_set, test_set]
    ])
    item_matrix = np.nan_to_num(dataset.items.toarray(), copy=True, nan=0)
    dist = pairwise_distances(item_matrix.T, metric=metric)
    items_train, item_test = [np.unique(cv_folds[i].row) for i in [0,1]]
    dist_lst = (
        np.ma.masked_equal(dist[items_train,:][:,item_test], 0.0, copy=True).min(), 
        np.ma.masked_equal(dist[items_train,:][:,items_train], 0.0, copy=True).min(), 
        np.ma.masked_equal(dist[item_test,:][:,item_test], 0.0, copy=True).min()
    )
    return cv_folds, dist_lst

def random_cv_split(dataset, cv_generator, metric="cosine"):
    '''
    Splits the data into training and testing datasets randomly for cross-validation.

    ...

    Parameters
    ----------
    dataset : stanscofi.Dataset
        dataset to split
    cv_generator : scikit-learn cross-validation index generator
        e.g. StratifiedKFold, KFold
    metric : str
        metric to consider to assess distance between training and testing sets. Should belong to [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’, ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘correlation’, ‘dice’, ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’]

    Returns
    ----------
    cv_folds : list of size nsplits of COO-array of shape (n_items, n_users)
        list of arrays which contain values in {0, 1} describing the unavailable and available user-item matchings in the training (resp. testing) set
    dist_lst : list of size nsplits of tuples of float of size 3
        for each fold, minimum nonzero distance between an element in the training and in the testing sets, resp. inside the training set, resp. inside the testing set
    '''
    X = np.column_stack((dataset.folds.row, dataset.folds.col)) ## indices for available ratings
    y = dataset.ratings.toarray()[dataset.folds.row,dataset.folds.col].ravel() ## values for available ratings
    cv_folds = [
        (
            indices_to_folds(train_index, X, dataset.folds.shape), 
            indices_to_folds(test_index, X, dataset.folds.shape)
        ) for train_index, test_index in cv_generator.split(X, y)]
    item_matrix = np.nan_to_num(dataset.items.toarray(), copy=True, nan=0)
    dist = pairwise_distances(item_matrix.T, metric=metric)
    dist_lst = [(
        np.ma.masked_equal(dist[np.unique(train_folds.row),:][:,np.unique(test_folds.row)], 0.0, copy=True).min(), 
        np.ma.masked_equal(dist[np.unique(train_folds.row),:][:,np.unique(train_folds.row)], 0.0, copy=True).min(), 
        np.ma.masked_equal(dist[np.unique(test_folds.row),:][:,np.unique(test_folds.row)], 0.0, copy=True).min()
    ) for train_folds, test_folds in cv_folds]
    return cv_folds, dist_lst

def weakly_correlated_split(dataset, test_size, early_stop=None, metric="cosine", random_state=1234, verbose=False):
    '''
    Splits the data into training and testing datasets with a low correlation among items, by applying a hierarchical clustering on the item feature matrix. NaNs in the item feature matrix are converted to 0. 

    ...

    Parameters
    ----------
    dataset : stanscofi.Dataset
        dataset to split
    test_size : float
        value between 0 and 1 (strictly) which indicates the maximum percentage of initial data (positive and negative ratings) being assigned to the test dataset
    early_stop : int or None
        positive integer, which stops the cluster number search after 3 tries yielding the same number; note that if early_stop is not None, then the property on test_size will not necessarily hold anymore
    metric : str
        metric to consider to perform hierarchical clustering on the dataset. Should belong to [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’, ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘correlation’, ‘dice’, ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’]
    random_state : int
        random seed 
    verbose : bool
        prints out information

    Returns
    ----------
    train_folds, test_folds : COO-array of shape (n_items, n_users)
        an array which contains values in {0, 1} describing the unavailable and available user-item matchings in the training (resp. testing) set
    dist_train_test, dist_train, dist_test : float
        minimum nonzero distance between an element in the training and in the testing sets, resp. inside the training set, resp. inside the testing set
    '''
    assert random_state > 0
    assert (early_stop is None) or (early_stop > 0)
    assert test_size > 0 and test_size < 1
    assert metric in ["cityblock", "cosine", "euclidean", "l1", "l2", "manhattan", "braycurtis", "canberra", "chebyshev", "correlation", "dice", "hamming", "jaccard", "kulsinski", "mahalanobis", "minkowski", "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener", "sokalsneath", "sqeuclidean", "yule"]
    np.random.seed(random_state)
    random.seed(random_state) 

    item_matrix = np.nan_to_num(dataset.items.toarray(), copy=True, nan=0)
    train_nset = int((1-test_size)*dataset.folds.data.shape[0])

    dist = pairwise_distances(item_matrix.T, metric=metric)
    Z = linkage(squareform(dist, checks=False), "average")
    select_nc, n_cluster_train = None, None
    l_nc, u_nc = 2, item_matrix.shape[1]
    count_sim, oldclnb = 0, None
    ## bisection to find the appropriate number of clusters, and where to split the data
    while (l_nc<u_nc):
        nc = (l_nc+u_nc)//2
        clusters = fcluster(Z, nc, criterion='maxclust', depth=2, R=None, monocrit=None)
        nratings_train = {len([x for x in dataset.folds.row if (clusters[x]<=c)]):c for c in range(1,len(np.unique(clusters))+1)}
        select_clust = np.max([k if (k<=train_nset) else -1 for k in nratings_train])
        #print("#training=%d\t#clust=%d\t#clusters=%d" % (train_nset, select_clust, nratings_train[select_clust]))
        #print(np.max([k if (k<select_clust) else -1 for k in nratings_train]), select_clust, np.min([k if (k>select_clust) else len(dataset.folds.row) for k in nratings_train]))
        cluster_size = nratings_train.get(select_clust, -1)
        if (verbose):
            print("<training_testing.traintest_validation_split> Find #clusters=%d in [%d, %d] (%d ~ %d?)" % (nc, l_nc, u_nc, select_clust, train_nset))
        if (select_clust==train_nset):
            break
        if (select_clust==oldclnb):
            count_sim += 1
            if ((early_stop is not None) and (count_sim>=early_stop)):
                break
        else:
            oldclnb = select_clust
        if (select_clust<train_nset):
            l_nc = nc+1
        else:
            u_nc = nc
    select_nc = nc
    cluster_size = cluster_size
    ## reproduces an old behavior which did not take into account the test_size parameter and maximizes the distance between training and testing sets
    #select_nc, cluster_size = 2, 0 
    item_labels = (fcluster(Z, select_nc, criterion='maxclust', depth=2, R=None, monocrit=None)>cluster_size+1).astype(int)+1

    ## COO-array of shape (n_items, n_users)
    train_folds = dataset.folds.toarray()
    train_folds[item_labels==2,:] = 0
    test_folds = dataset.folds.toarray()-train_folds

    dist_lst = (
        np.ma.masked_equal(dist[item_labels==1,:][:,item_labels==2], 0.0, copy=True).min(), 
        np.ma.masked_equal(dist[item_labels==1,:][:,item_labels==1], 0.0, copy=True).min(), 
        np.ma.masked_equal(dist[item_labels==2,:][:,item_labels==2], 0.0, copy=True).min()
    )

    return (coo_array(train_folds), coo_array(test_folds)), dist_lst

##############################
# Common training procedure  #
##############################

def cv_training(template, params, train_dataset, nsplits, metric, k=1, beta=1, threshold=0, test_size=0.2, dist_type="cosine", cv_type=["random","weakly_correlated"][0], early_stop=2, njobs=1, random_state=1234, show_plots=False, verbose=False):
    '''
    Trains a model on a dataset using cross-validation and custom metrics using sklearn.model_selection.StratifiedKFold

    ...

    Parameters
    ----------
    template : stanscofi.BasicModel or subclass
        type of model to train
    params : dict
        dictionary of parameters to initialize the model
    train_dataset : stanscofi.Dataset
        dataset to train upon
    nsplits : int
        number of cross-validation steps
    metric : str 
        metric to optimize the model upon. Implemented metrics are in validation.py
    k : int (default: 1)
        Argument of the metric to optimize. Implemented metrics are in validation.py
    beta : float (default: 1)
        Argument of the metric to optimize. Implemented metrics are in validation.py
    threshold : float (default: 0)
        decision threshold
    test_size : float (default: 0.2)
        percentage of testing set (if cv_type="weakly_correlated")
    dist_type : str (default: "cosine")
        type of metric for splitting (if cv_type="weakly_correlated")
    cv_type : str (default: "random")
        type of split to apply to the dataset. Can either be "random" or "weakly_correlated"
    early_stop : int or None
        positive integer, which stops the cluster number search after 3 tries yielding the same number; note that if early_stop is not None, then the property on test_size will not necessarily hold anymore
    njobs : int (default: 1)
        number of jobs to run in parallel. Should be lower than nsplits-1
    random_state : int (default: 1234)
        random seed
    show_plots : bool (default: False)
        shows the validation plots at each cross-validation step
    verbose : bool (default: False)
        prints out information

    Returns
    ----------
    results : dict
        a dictionary which contains 
            "models" : list of subinstances of stanscofi.models.BasicModel of length nsplits
                all trained models
            "train_metric" : list of floats of length nsplits
                all metrics on training sets
            "test_metric" : list of floats of length nsplits
                all metrics on testing sets
            "cv_folds" : list of COO-array of shape (n_items, n_users) of length nsplits
                the training and testing folds for each split
    '''
    assert cv_type in ["random", "weakly_correlated"]
    assert random_state > 0
    assert test_size >0 and test_size<1
    assert nsplits > 1
    assert njobs in range(nsplits)
    assert metric in metrics_list

    np.random.seed(random_state)
    random.seed(random_state)
    parallel = Parallel(n_jobs=njobs, verbose=verbose)

    if (cv_type=="random"):
        cv_generator = StratifiedKFold(n_splits=nsplits, shuffle=True, random_state=random_state)
        cv_folds, _ = random_cv_split(train_dataset, cv_generator)
    else:
        seeds = np.random.choice(range(int(1e8)), size=nsplits)
        cv_folds = [weakly_correlated_split(train_dataset, test_size, metric=dist_type, early_stop=early_stop, random_state=sd)[0] for sd in seeds]

    def single_run(ncv, template, params, metric, tfolds, sfolds):
        model = template(params)
        tdataset = train_dataset.subset(tfolds)
        sdataset = train_dataset.subset(sfolds)
        model.fit(tdataset, seed=1234)
        scores_train = model.predict_proba(tdataset)
        predictions_train = model.predict(scores_train, threshold)
        #print((tdataset.folds.data.shape[0], predictions_train.data.shape[0], scores_train.data.shape[0]))
        metrics_train, _ = compute_metrics(scores_train, predictions_train, tdataset, metrics=[metric], verbose=verbose)
        scores_test = model.predict_proba(sdataset)
        predictions_test = model.predict(scores_test, threshold)
        #print((sdataset.folds.data.shape[0], predictions_test.data.shape[0], scores_test.data.shape[0]))
        metrics_test, plot_args = compute_metrics(scores_test, predictions_test, sdataset, metrics=[metric], k=k, beta=beta, verbose=verbose)
        if (show_plots):
            plot_metrics(**plot_args, figsize=(10,10), model_name="%s on %s (cv%d)\n" % (model.name, train_dataset.name, ncv+1))
        metric_train, metric_test = metrics_train.loc[metric][metrics_train.columns[0]], metrics_test.loc[metric][metrics_test.columns[0]]
        if (verbose):
            print("Crossvalidation step #%d/%d (train %s %f, test %s %f)" % (ncv+1,nsplits,metric,metric_train,metric,metric_test))
        return model, metric_train, metric_test

    results = parallel(
        delayed(single_run)(
            ncv, deepcopy(template), params, metric, tfolds, sfolds
        ) for ncv, (tfolds, sfolds) in enumerate(cv_folds)
    )
    return {"models": [r[0] for r in results], "train_metric": [r[1] for r in results], "test_metric": [r[2] for r in results], "cv_folds": cv_folds}

##############################
# Hyperparameter search      #
##############################

def grid_search(search_params, template, params, train_dataset, nsplits, metric, k=1, beta=1, threshold=0, test_size=0.2, dist_type="cosine", cv_type=["random","weakly_correlated"][0], early_stop=2, njobs=1, random_state=1234, show_plots=False, verbose=False):
    '''
    Grid-search over hyperparameters, iteratively optimizing over one parameter at a time, and internally calling cv_training.

    ...

    Parameters
    ----------
    search_params : dict
        a dictionary which contains as keys the hyperparameter names and as values the corresponding intervals to explore during the grid-search
    template : stanscofi.BasicModel or subclass
        type of model to train
    params : dict
        dictionary of parameters to initialize the model
    train_dataset : stanscofi.Dataset
        dataset to train upon
    metric : str 
        metric to optimize the model upon. Implemented metrics are in validation.py
    k : int (default: 1)
        Argument of the metric to optimize. Implemented metrics are in validation.py
    beta : float (default: 1)
        Argument of the metric to optimize. Implemented metrics are in validation.py
    threshold : float (default: 0)
        decision threshold
    test_size : float (default: 0.2)
        percentage of testing set (if cv_type="weakly_correlated")
    dist_type : str (default: "cosine")
        type of metric for splitting (if cv_type="weakly_correlated")
    cv_type : str (default: "random")
        type of split to apply to the dataset. Can either be "random" or "weakly_correlated"
    njobs : int (default: 1)
        number of jobs to run in parallel. Should be lower than nsplits-1
    random_state : int (default: 1234)
        random seed
    show_plots : bool (default: False)
        shows the validation plots at each cross-validation step
    verbose : bool (default: False)
        prints out information

    Returns
    ----------
    best_params : dict
        a dictionary which contains as keys the hyperparameter names and as values the best values obtained across all grid-search steps
    best_model : subinstance of stanscofi.models.BasicModel
        the best trained model associated with the best parameters
    metrics : dict        
        a dictionary which contains 
            "train_metric" : float 
                the metric on the training set on the best crossvalidation split for the best set of parameters
            "test_metric" : float
                the metric on the testing set on the best crossvalidation split for the best set of parameters
    '''
    best_params, best_model, best_test_metric, best_train_metric = {}, None, -float("inf"), 0
    for param in search_params:
        for param_val in search_params[param]:
            params_ = params.copy()
            params_.update(best_params)
            params_.update({param: param_val})
            results = cv_training(template, params, train_dataset, nsplits, metric=metric, threshold=threshold, test_size=test_size, dist_type=dist_type, cv_type=cv_type, early_stop=early_stop, njobs=njobs, random_state=random_state, show_plots=show_plots, verbose=verbose)
            if (verbose):
                print("<training_testing.grid_search> [%s=%s] best %s on Test %f (Train %f)" % (param, str(param_val), metric, np.max(results["test_metric"]), results["train_metric"][np.argmax(results["test_metric"])]))
            if (np.max(results["test_metric"])>best_test_metric):
                best_params.update(params_)
                best_model = deepcopy(results["models"][np.argmax(results["test_metric"])])
                best_test_metric = np.max(results["test_metric"])
                best_train_metric = results["train_metric"][np.argmax(results["test_metric"])]
    return best_params, best_model, {"test_metric": best_test_metric, "train_metric": best_train_metric}