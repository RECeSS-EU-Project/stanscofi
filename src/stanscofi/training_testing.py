#coding: utf-8

import pandas as pd
import numpy as np
from copy import deepcopy
from joblib import Parallel, delayed
from tqdm import tqdm

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import StratifiedKFold

import stanscofi.validation

##############################
# Weak correlation splitting #
##############################

def print_folds(folds, dataset, fold_name="Fold"):
    '''
    Prints out information about a folds: number of unique drugs resp., diseases, number of considered ratings, proportion of positive and negative ratings

    ...

    Parameters
    ----------
    folds : array-like of shape (n_ratings, 3)
        a matrix which contains the user indices (column 1), the item indices (column 2) and the class for the corresponding (user, item) pair (value in {-1, 0, 1} in column 3)
    dataset : stanscofi.Dataset
        dataset which can be subsetted by folds
    fold_name : str
        name given to the fold
    '''
    if (len(folds)==0):
        print("<training_testing.print_folds> Fold is empty.")
        return None
    assert folds.shape[1]==3
    assert np.max(folds[:,0])<=dataset.ratings_mat.shape[1]
    assert np.max(folds[:,1])<=dataset.ratings_mat.shape[0]
    assert ((folds[:,-1]==1)|(folds[:,-1]==-1)|(folds[:,-1]==0)).all()
    name, ndrugs, ndiseases = [fold_name]+[np.unique(folds[:,x]).shape[0] for x in [1,0]]
    nratings, percratings = [folds.shape[0], folds.shape[0]*100/dataset.ratings.shape[0]]
    args = [name, ndrugs, ndiseases, nratings, percratings]
    print("\n<training_testing.print_folds> %s: %d unique drugs and %d unique diseases and %d ratings (%.2f perc.)" % tuple(args))
    counts = [ndrugs*ndiseases-folds.shape[0]]+[np.sum(folds[:,-1]==v) for v in [1,-1]]
    print(pd.DataFrame(counts, index=[0,1,-1], columns=["counts"]).T)

def traintest_validation_split(dataset, test_size, early_stop=None, metric="cosine", disjoint_users=False, random_state=1234, verbose=False, print_dists=False):
    '''
    Splits the data into training and testing datasets with a low correlation

    ...

    Parameters
    ----------
    dataset : stanscofi.Dataset
        dataset to split
    test_size : float
        value between 0 and 1 (strictly) which indicates the maximum percentage of initial data (known ratings) being assigned to the test dataset
    early_stop : int or None
        positive integer, which stops the cluster number search after 3 tries yielding the same number; note that if early_stop is not None, then the property on test_size will not necessarily hold anymore
    metric : str
        metric to consider to perform hierarchical clustering on the dataset. Should belong to [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’, ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘correlation’, ‘dice’, ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’]
    disjoint_users : bool
        whether to split the data in four sets such that two of these sets have a disjoint set of diseases, that is,
        if disjoint_users=False (default): Splits the initial dataset into two parts with the following properties
            - The sets of users in both parts are disjoint
            - The sets of items in training+testing sets and validation sets are weakly correlated
              (by applying a hierarchical clustering on the item feature matrix, NaN are converted to 0)
        if disjoint_users=True: Splits the initial dataset into four parts with the following properties
            - The sets of users in train_folds and test_folds, resp. val_folds and supp_folds, 
              resp. val_folds and train_folds, resp. test_folds and supp_folds are disjoint
            - The sets of items are weakly correlated between
               train_folds and test_folds/val_folds, test_folds and train_folds/supp_folds
               and disjoint train_folds and test_folds, resp. val_folds and supp_folds, 
              resp. val_folds and test_folds, resp. train_folds and supp_folds
              (by applying a hierarchical clustering on the item feature matrix, NaN are converted to 0)
    random_state : int
        random seed 
    verbose : bool
        prints out information
    print_dists : bool
        prints distances between returned folds

    Returns
    ----------
    train_folds, test_folds, val_folds, supp_folds : array-like of shape (n_ratings, 3)
        a matrix which contains the user indices (column 1), the item indices (column 2) and the class for the corresponding (user, item) pair (value in {-1, 0, 1} in column 3)
    '''
    assert random_state > 0
    assert (early_stop is None) or (early_stop > 0)
    assert test_size > 0 and test_size < 1
    assert metric in ["cityblock", "cosine", "euclidean", "l1", "l2", "manhattan", "braycurtis", "canberra", "chebyshev", "correlation", "dice", "hamming", "jaccard", "kulsinski", "mahalanobis", "minkowski", "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener", "sokalsneath", "sqeuclidean", "yule"]
    np.random.seed(random_state)
    item_matrix = np.nan_to_num(dataset.items, copy=True)
    ratings = dataset.ratings.copy()
    train_nset = int((1-test_size)*ratings.shape[0])

    dist = pairwise_distances(item_matrix.T,metric=metric)
    dist_ = squareform(dist, checks=False)
    Z = linkage(dist_, "average")
    select_nc, n_cluster_train = None, None
    l_nc, u_nc = 2, item_matrix.shape[1]
    count_sim, oldclnb = 0, None
    ## bisection to find the appropriate number of clusters, and where to split the data
    while (l_nc<u_nc):
        nc = (l_nc+u_nc)//2
        clusters = fcluster(Z, nc, criterion='maxclust', depth=2, R=None, monocrit=None)
        nratings_train = {ratings[np.array([clusters[x]<=c for x in ratings[:,1].astype(int).tolist()]),:].shape[0]:c for c in range(1,len(np.unique(clusters))+1)}
        select_clust = np.max([k if (k<=train_nset) else -1 for k in nratings_train])
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
    #select_nc, cluster_size = 2, 0 ## reproduce an old behavior which did not take into account the test_size parameter
    item_labels = (fcluster(Z, select_nc, criterion='maxclust', depth=2, R=None, monocrit=None)>cluster_size+1).astype(int)+1

    item_cluster = np.array([item_labels[item] for item in ratings[:,1].astype(int).tolist()])
    train_set = ratings[item_cluster==1,:]
    test_set = ratings[item_cluster==2,:]

    f = lambda x : np.min(x[x!=0]) if (x[x!=0].shape[0]>0) else 0
    get_dist = lambda s1, s2 : f(dist[np.unique(s1[:,1]).tolist(),:][:,np.unique(s2[:,1]).tolist()])

    if (not disjoint_users):

        if (print_dists): 
            print("<training_testing.traintest_validation_split> The sets of items/drugs in the training/testing datasets are disjoint.")
            print("<training_testing.traintest_validation_split> The sets of users/diseases in the training/testing datasets are *not* disjoint.")
            print("<training_testing.traintest_validation_split> Distances:")
            cols = ["Train set","Test set"]
            sets = [train_set, test_set]
            dists = pd.DataFrame([[get_dist(c1,c2) for c1 in sets] for c2 in sets], index=cols, columns=cols)
            print(dists)

        return train_set.astype(int), test_set.astype(int), np.array([]), np.array([])

    user_cluster = np.array([np.sum(np.unique(item_cluster[ratings[ratings[:,0]==x,1].astype(int).tolist()])) for x in ratings[:,0].astype(int).tolist()])
    train_set = ratings[user_cluster==1,:]
    test_set = ratings[user_cluster==2,:]
    validation_set = ratings[user_cluster==3,:]

    ## Greedy approach to avoid computational explosion
    ## Assign disease to the set for which the similarity with the newly added drugs is highest
    val_users = np.unique(validation_set[:,0]).flatten().tolist()
    val1_set = np.array([])
    val2_set = np.array([])
    get_subset = lambda cl : validation_set[(validation_set[:,0].flatten()==val_user)&(item_cluster[validation_set[:,1].flatten().tolist()]==cl),:]
    for val_user in val_users:
        valcl1, valcl2 = get_subset(1), get_subset(2)
        #assert validation_set[validation_set[:,0].flatten()==val_user,:].shape[0]==np.sum([X.shape[0] for X in [valcl1,valcl2]])
        train_set_ = np.concatenate((train_set, valcl1), axis=0)
        test_set_ = np.concatenate((test_set, valcl2), axis=0)
        #assert train_set.shape[0]+test_set.shape[0]+validation_set[validation_set[:,0].flatten()==val_user,:].shape[0]==np.sum([X.shape[0] for X in [train_set_,test_set_]])
        dist_train = get_dist(train_set_, test_set)
        dist_test = get_dist(train_set, test_set_)
        #print((dist_test, dist_train, test_set.shape[0], valcl2.shape[0], train_set.shape[0], valcl1.shape[0]))
        if (np.isclose(dist_test,dist_train)): # if equality
                weights = [x.shape[0]/(train_set.shape[0]+test_set.shape[0]) for x in [train_set, test_set]]
                dist_test += np.random.choice([-1,1], p=weights, size=1)
        if (dist_test<dist_train):
                test_set = np.copy(test_set_)
                if (len(val1_set)==0):
                    val1_set = np.copy(valcl1)
                else:
                    val1_set = np.concatenate((val1_set,valcl1), axis=0)
        else:
                train_set = np.copy(train_set_)
                if (len(val2_set)==0):
                    val2_set = np.copy(valcl2)
                else:
                    val2_set = np.concatenate((val2_set,valcl2), axis=0)

    if (print_dists): 
        print("<training_testing.traintest_validation_split> The sets of items/drugs in the training/testing datasets, in the validation/supplementary datasets, in the validation/testing and in the supplementary/training datasets are disjoint.")
        print("<training_testing.traintest_validation_split> The sets of items/drugs in the training/testing datasets, in the validation/supplementary datasets, in the validation/training and in the supplementary/testing datasets are disjoint.")
        print("<training_testing.traintest_validation_split> Distances")
        cols = ["Train set","Test set","Val set","Supp set"]
        sets = [train_set, test_set, val1_set, val2_set]
        dists = pd.DataFrame([[np.nan if ((len(c1)==0) or (len(c2)==0)) else get_dist(c1,c2) for c1 in sets] for c2 in sets], index=cols, columns=cols)
        print(dists)

    return train_set.astype(int), test_set.astype(int), val1_set.astype(int), val2_set.astype(int)

##############################
# Common training procedure  #
##############################

def cv_training(template, params, train_dataset, metric="AUC", beta=1, njobs=1, nsplits=5, random_state=1234, show_plots=True, verbose=False):
    '''
    Trains a model on a dataset using cross-validation using sklearn.model_selection.StratifiedKFold

    ...

    Parameters
    ----------
    template : stanscofi.BasicModel or subclass
        type of model to train
    params : dict
        dictionary of parameters to initialize the model
    train_dataset : stanscofi.Dataset
        dataset to train upon
    metric : str
        metric to optimize the model upon. Should belong to ["AUC", "F_%.1f" % <beta>] where <beta> is a parameter of cv_training
    beta : float
        beta for the computation of a F_beta score
    njobs : int
        number of jobs to run in parallel. Should be lower than nsplits-1
    nsplits : int
        number of cross-validation steps
    random_state : int
        random seed
    show_plots : bool
        shows the validation plots at each cross-validation step
    verbose : bool
        prints out information

    Returns
    ----------
    best_estimator : dict
        a dictionary which contains 
            "test_"+metric : float
                the best metric obtained across all validations on the testing set
            "train_"+metric : float
                the metric obtained on the training set in the best cross-validation step
            "model_params" : dict
                the parameters of the best performing model
            "cv_folds" : tuple of array-like of shape (n_ratings, 3)
                the corresponding training and testing folds for the best cross-validation step, where a fold is a matrix which contains the user indices (column 1), the item indices (column 2) and the class for the corresponding (user, item) pair (value in {-1, 0, 1} in column 3)
    '''
    assert beta > 0
    assert random_state > 0
    assert nsplits > 1
    assert metric in ["AUC", "F_%.1f" % beta]
    assert njobs in range(nsplits)
    cv_generator = StratifiedKFold(n_splits=nsplits, shuffle=True, random_state=random_state)
    Nitems, Nusers = len(np.unique(train_dataset.ratings[:,0])), len(np.unique(train_dataset.ratings[:,1]))
    grid = np.indices((Nitems,Nusers))
    full_list = np.zeros((Nitems*Nusers, 3))
    full_list[:,0] = grid[0].flatten()
    full_list[:,1] = grid[1].flatten()
    full_list[:,2] = [train_dataset.ratings_mat[j,i] for i,j in full_list[:,:2].astype(int).tolist()]
    full_list = full_list.astype(int)
    cv_folds = cv_generator.split(full_list[:,:2], np.ravel(full_list[:,2]))
    best_estimator, best_metric = {}, -float("inf")
    def single_run(ncv, tfolds, sfolds, full_list):
        if (verbose):
            print("Crossvalidation step #%d/%d" % (ncv+1,nsplits))
        model = template(params)
        tdataset = train_dataset.get_folds(full_list[tfolds,:])
        sdataset = train_dataset.get_folds(full_list[sfolds,:])
        model.fit(tdataset)
        scores_train = model.predict(tdataset)
        predictions_train = model.classify(scores_train)
        metrics_train, _ = stanscofi.validation.compute_metrics(scores_train, predictions_train, tdataset, beta=beta, verbose=verbose)
        scores_test = model.predict(sdataset)
        predictions_test = model.classify(scores_test)
        metrics_test, plot_args = stanscofi.validation.compute_metrics(scores_test, predictions_test, sdataset, beta=beta, verbose=verbose)
        if (show_plots):
            stanscofi.validation.plot_metrics(**plot_args, figsize=(10,10), model_name="%s on %s (cv%d)\n" % (model.name, train_dataset.name, ncv+1))
        return metrics_train.loc[metric][metrics_train.columns[0]], metrics_test.loc[metric][metrics_test.columns[0]], [tfolds, sfolds], params
    if (njobs==1):
        metrics_list = [single_run(ncv, tfolds, sfolds, full_list) for ncv, [tfolds, sfolds] in tqdm(enumerate(cv_folds))]
    else:
        metrics_list = Parallel(n_jobs=njobs, backend='loky')(delayed(single_run)(ncv, tfolds, sfolds, full_list) for ncv, [tfolds, sfolds] in tqdm(enumerate(cv_folds)))
    metric_test = [m for _, m, _, _ in metrics_list]
    best_id = np.argmax(metric_test)
    best_metric = metric_test[best_id]
    best_train_metric = metrics_list[best_id][0]
    best_model = metrics_list[best_id][3]
    best_folds = [full_list[ids,:] for ids in metrics_list[best_id][2]]
    best_estimator = {"test_"+metric: best_metric, "train_"+metric: best_train_metric, "model_params": best_model, "cv_folds": best_folds}
    return best_estimator

def grid_search(search_params, template, params, train_dataset, metric="AUC", njobs=1, nsplits=5, random_state=1234, show_plots=True, verbose=False):
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
        metric to optimize the model upon. Should belong to ["AUC", "F_%.1f" % <beta>] where <beta> is a parameter of cv_training
    beta : float
        beta for the computation of a F_beta score
    njobs : int
        number of jobs to run in parallel. Should be lower than nsplits-1
    nsplits : int
        number of cross-validation steps
    random_state : int
        random seed
    show_plots : bool
        shows the validation plots at each cross-validation step
    verbose : bool
        prints out information

    Returns
    ----------
    best_params : dict
        a dictionary which contains as keys the hyperparameter names and as values the best values obtained across all grid-search steps
    best_estimator : dict
        a dictionary which contains 
            "test_"+metric : float
                the best metric obtained across all validations on the testing set
            "train_"+metric : float
                the metric obtained on the training set in the best cross-validation step
            "model_params" : dict
                the parameters of the best performing model
            "cv_folds" : tuple of array-like of shape (n_ratings, 3)
                the corresponding training and testing folds for the best cross-validation step, where a fold is a matrix which contains the user indices (column 1), the item indices (column 2) and the class for the corresponding (user, item) pair (value in {-1, 0, 1} in column 3)
    '''
    best_params, best_model, best_metric = {}, None, -float("inf")
    for param in search_params:
        for param_val in search_params[param]:
            params_ = params.copy()
            params_.update(best_params)
            params_.update({param: param_val})
            best_estimator = cv_training(template, params_, train_dataset, metric=metric, njobs=njobs, nsplits=nsplits, random_state=random_state, show_plots=show_plots, verbose=verbose)
            if (verbose):
                print("<training_testing.grid_search> [%s=%s] %s on Test %f (Train %f)" % (param, str(param_val), metric, best_estimator["test_"+metric], best_estimator["train_"+metric]))
            if (best_estimator["test_"+metric]>best_metric):
                best_params.update(params_)
                best_model = deepcopy(best_estimator)
                best_metric = best_estimator["test_"+metric]
    return best_params, best_model