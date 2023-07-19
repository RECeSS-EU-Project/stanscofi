#coding: utf-8

import pandas as pd
import numpy as np
from scipy.sparse import coo_array, csr_array
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from sklearn.decomposition import PCA
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*he 'nopython' keyword argument was not supplied to the 'numba.jit' decorator.*")
    import umap

from .preprocessing import meanimputation_standardize

def generate_dummy_dataset(npositive, nnegative, nfeatures, mean, std, random_state=12454):
    '''
    Creates a dummy dataset where the positive and negative (item, user) pairs are arbitrarily similar. 

    Each of the nfeatures features for (item, user) pair feature vectors associated with positive ratings are drawn from a Gaussian distribution of mean mean and standard deviation std, whereas those for negative ratings are drawn from from a Gaussian distribution of mean -mean and standard deviation std. User and item feature matrices of shape (nfeatures//2, npositive+nnegative) are generated, which are the concatenation of npositive positive and nnegative negative pair feature vectors generated from Gaussian distributions. Thus there are npositive^2 positive ratings (each "positive" user with a "positive" item), nnegative^2 negative ratings (idem), and the remainder is unknown (that is, (npositive+nnegative)^2-npositive^2-nnegative^2 ratings).

    ...

    Parameters
    ----------
    npositive : int
        number of positive items/users
    nnegative : int
        number of negative items/users
    nfeatures : int
        number of item/user features
    mean : float
        mean of generating Gaussian distributions
    std : float
        standard deviation of generating Gaussian distributions

    Returns
    ----------
    ratings : array-like of shape (n_items, n_users)
        a matrix which contains values in {-1, 0, 1} describing the known and unknown user-item matchings
    users : array-like of shape (n_item_features, n_items)
        a list of the item feature names in the order of column indices in ratings_mat
    items : array-like of shape (n_user_features, n_users)
        a list of the item feature names in the order of column indices in ratings_mat
    '''
    assert nfeatures%2==0
    np.random.seed(random_state)
    ## Generate feature matrices
    nusers = nitems = npositive+nnegative
    positive = np.random.normal(mean,std,size=(nfeatures,npositive))
    negative = np.random.normal(-mean,std,size=(nfeatures,nnegative))
    users = np.concatenate((positive, negative), axis=1)[:nfeatures//2,:]
    items = np.concatenate((positive, negative), axis=1)[nfeatures//2:,:]
    ## Generate ratings
    ratings = np.zeros((nitems, nusers))
    ratings[:npositive,:npositive] = 1
    ratings[npositive:,npositive:] = -1
    ## Input to stanscofi
    return {"ratings": coo_array(ratings), "users": users, "items": items}

class Dataset(object):
    '''
    A class used to encode a drug repurposing dataset (items are drugs, users are diseases)

    ...

    Parameters
    ----------
    ratings : array-like of shape (n_items, n_users)
        an array which contains values in {-1, 0, 1, np.nan} describing the negative, unlabelled, positive, unavailable user-item matchings
    items : array-like of shape (n_item_features, n_items)
        an array which contains the item feature vectors
    users : array-like of shape (n_user_features, n_users)
        an array which contains the user feature vectors
    same_item_user_features : bool (default: False)
        whether the item and user features are the same (optional)
    name : str
        name of the dataset (optional)

    Attributes
    ----------
    name : str
        name of the dataset 
    ratings : COO-array of shape (n_items, n_users)
        an array which contains values in {-1, 0, 1} describing the negative, unlabelled/unavailable, positive user-item matchings
    folds : COO-array of shape (n_items, n_users)
        an array which contains values in {0, 1} describing the unavailable and available user-item matchings in ratings
    items : COO-array of shape (n_item_features, n_items)
        an array which contains the user feature vectors (NaN for missing features)
    users : COO-array of shape (n_user_features, n_users)
        an array which contains the item feature vectors (NaN for missing features)
    item_list : list of str
        a list of the item names in the order of row indices in ratings_mat
    user_list : list of str
        a list of the user names in the order of column indices in ratings_mat
    item_features : list of str
        a list of the item feature names in the order of column indices in ratings_mat
    user_features : list of str
        a list of the user feature names in the order of column indices in ratings_mat
    same_item_user_features : bool
        whether the item and user features are the same
    nusers : int
        number of users
    nitems : int
        number of items
    nuser_features : int
        number of user features
    nitem_features : int
        number of item features

    Methods
    -------
    __init__(ratings=None, users=None, items=None, same_item_user_features=False, name="dataset")
        Initialize the Dataset object and creates all attributes
    summary(sep="-"*70)
        Prints out the characteristics of the drug repurposing dataset
    visualize(withzeros=False, X=None, y=None, figsize=(5,5), fontsize=20, dimred_args={}, predictions=None, use_ratings=False, random_state=1234, show_errors=False, verbose=False)
        Plots datapoints in the dataset annotated by the ground truth or predicted ratings
    subset(folds, subset_name="subset")
        Creates a subset of the dataset based on the folds given as input
    '''
    def __init__(self, ratings=None, users=None, items=None, same_item_user_features=False, name="dataset"):
        '''
        Creates an instance of stanscofi.Dataset

        ...

        Parameters
        ----------
        ratings : array-like of shape (n_items, n_users)
            an array which contains values in {-1, 0, 1, np.nan} describing the negative, unlabelled, positive, unavailable user-item matchings
        items : array-like of shape (n_item_features, n_items)
            an array which contains the item feature vectors
        users : array-like of shape (n_user_features, n_users)
            an array which contains the user feature vectors
        same_item_user_features : bool (default: False)
            whether the item and user features are the same (optional)
        name : str
            name of the dataset (optional)
        '''
        assert ratings is not None
        assert users is not None and users.shape[1]==ratings.shape[1]
        assert items is not None and items.shape[1]==ratings.shape[0]
        ## get metadata
        if (str(type(ratings))=="<class 'pandas.core.frame.DataFrame'>"):
            self.item_list = [str(x) for x in ratings.index]
            self.user_list = [str(x) for x in ratings.columns]
            ratings_ = ratings.values
        else:
            self.item_list = [str(x) for x in range(ratings.shape[0])]
            self.user_list = [str(x) for x in range(ratings.shape[1])]
            if (str(type(ratings))=="<class 'scipy.sparse._arrays.coo_array'>"):
                ratings_ = ratings.toarray()
            else:
                ratings_ = ratings.copy()
            ratings_ = ratings_.copy()
        if (str(type(users))=="<class 'pandas.core.frame.DataFrame'>"):
            users = users[self.user_list]
            self.user_features = [str(x) for x in users.index]
            users_ = users.values
        else:
            self.user_features = [str(x) for x in range(users.shape[0])]
            users_ = users.copy()
        if (str(type(items))=="<class 'pandas.core.frame.DataFrame'>"):
            items = items[self.item_list]
            self.item_features = [str(x) for x in items.index]
            items_ = items.values
        else:
            self.item_features = [str(x) for x in range(items.shape[0])]
            items_ = items.copy()
        self.same_item_user_features = same_item_user_features
        if (self.same_item_user_features):
            features = list(set(self.item_features).intersection(set(self.user_features)))
            assert len(features)>0
            self.user_features = features
            self.item_features = features
            users = users.loc[features]
            items = items.loc[features]
        ## format
        self.name = name
        self.ratings = coo_array(np.nan_to_num(ratings_, copy=True, nan=0))
        ids = np.argwhere(~np.isnan(ratings_))
        row = ids[:,0].ravel()
        col = ids[:,1].ravel()
        data = [1]*ids.shape[0]
        self.folds = coo_array((data, (row, col)), shape=ratings_.shape)
        self.users = coo_array(users_)
        self.items = coo_array(items_)
        self.nusers = self.users.shape[1]
        self.nitems = self.items.shape[1]
        self.nuser_features = self.users.shape[0]
        self.nitem_features = self.items.shape[0]

    def summary(self, sep="-"*70):
        '''
        Prints out a summary of the contents of a stanscofi.Dataset: the number of items, users, the number of positive, negative, unlabeled, unavailable matchings, the sparsity number, and the shape and percentage of missing values in the item and user feature matrices

        ...

        Parameters
        ----------
        sep : str
            separator for pretty printing
        ...

        Returns
        ----------
        ndrugs : int
            number of drugs
        ndiseases : int
            number of diseases
        ndrugs_known : int
            number of drugs with at least one known (positive or negative) rating
        ndiseases_known : int
            number of diseases with at least one known (positive or negative) rating
        npositive : int
            number of positive ratings
        nnegative : int
            number of negative ratings
        nunlabeled_unavailable : int
            number of unlabeled or unavailable ratings
        nunavailable : int
            number of unavailable ratings
        sparsity : float
            percentage of known ratings
        sparsity_known : float
            percentage of known ratings among drugs and diseases with at least one known rating
        ndrug_features : int
            number of drug features
        missing_drug_features : float
            percentage of missing drug feature values
        ndisease_features : int
            number of disease features
        missing_disease_features : float
            percentage of missing disease feature values
        '''
        print(sep)
        print("* Rating matrix: %d drugs x %d diseases" % (self.nitems, self.nusers))
        restricted_ratings = self.ratings.toarray()[np.abs(self.ratings).sum(axis=1)>0,:]
        restricted_ratings = restricted_ratings[:,np.abs(restricted_ratings).sum(axis=0)>0]
        print("Including %d drugs and %d diseases involved in at least one positive/negative rating" % restricted_ratings.shape)
        print("%d positive, %d negative, %d unlabeled (including %d unavailable) drug-disease ratings" % ((self.ratings==1).sum(), (self.ratings==-1).sum(), np.prod(self.ratings.shape)-self.ratings.getnnz(), np.prod(self.folds.shape)-self.folds.getnnz()))
        print("Sparsity: %.2f percent (on drugs/diseases with at least one known rating %.2f)" % ((self.ratings!=0).mean()*100, (restricted_ratings!=0).mean()*100))
        print(sep[:len(sep)//2])
        print("* Feature matrices:")
        if (self.items.shape[0]>0):
            print("#Drug features: %d\tTotal #Drugs: %d" % (self.items.shape))
            print("Missing features: %.2f percent" % (np.isnan(self.items.toarray()).mean()*100))
        if (self.users.shape[0]>0):
            print("#Disease features: %d\tTotal #Disease: %d" % (self.users.shape))
            print("Missing features: %.2f percent" % (np.isnan(self.users.toarray()).mean()*100))
        if (self.users.shape[0]+self.items.shape[0]==0):
            print("No feature matrices.")
        print(sep+"\n")
        return self.nitems, self.nusers, restricted_ratings.shape[0], restricted_ratings.shape[1], (self.ratings==1).sum(), (self.ratings==-1).sum(), np.prod(self.ratings.shape)-self.ratings.getnnz(), np.prod(self.folds.shape)-self.folds.getnnz(), (self.ratings!=0).mean()*100, (restricted_ratings!=0).mean()*100, self.items.shape[0], np.isnan(self.items.toarray()).mean()*100, self.users.shape[0], np.isnan(self.users.toarray()).mean()*100

    def visualize(self, withzeros=False, X=None, y=None, metric="euclidean", figsize=(5,5), fontsize=20, dimred_args={}, predictions=None, use_ratings=False, random_state=1234, show_errors=False, verbose=False):
        '''
        Plots a representation of the datapoints in a stanscofi.Dataset which is annotated either by the ground truth labels or the predicted labels. The representation is the plot of the datapoints according to the first two Principal Components, or the first two dimensions in UMAP, if the feature matrices can be converted into a (n_ratings, n_features) shaped matrix where n_features>1, else it plots a heatmap with the values in the matrix for each rating pair. 

        In the legend, ground truth labels are denoted with brackets: e.g., [0] (unknown), [1] (positive) and [-1] (negative); predicted ratings are denoted by "pos" (positive) and "neg" (negative); correct (resp., incorrect) predictions are denoted by "correct", resp. "error"

        ...

        Parameters
        ----------
        withzeros : bool
            boolean to assess whether (user, item) unknown matchings should also be plotted; if withzeros=False, then only (item, user) pairs associated with known matchings will be plotted (but the unknown matching datapoints will still be used to compute the dimensionality reduction); otherwise, all pairs will be plotted
        X : array-like of shape (n_ratings, n_features) or None
            (item, user) pair feature matrix
        y : array-like of shape (n_ratings, ) or None
            response vector for each (item, user) pair in X; necessarily X should not be None if y is not None, and vice versa; setting X and y automatically overrides the other behaviors of this function
        metric : str
            metric to consider to perform hierarchical clustering on the dataset. Should belong to [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’, ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘correlation’, ‘dice’, ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’]
        figsize : tuple of size 2
            width and height of the figure
        fontsize : int
            size of the legend, title and labels of the figure
        dimred_args : dict
            dictionary which lists the parameters to the dimensionality reduction method (either PCA, by default, or UMAP, if parameter "n_neighbors" is provided)
        predictions : array-like of shape (n_ratings, 3) or None
            a matrix which contains the user indices (column 1), the item indices (column 2) and the class for the corresponding (user, item) pair (value in {-1, 0, 1} in column 3); if predictions=None, then the ground truth ratings will be used to color datapoints, otherwise, the predicted ratings will be used
        use_ratings : bool
            if set to True, use the ratings in the dataset as predictions (for debugging purposes)
        random_state : int
            random seed
        show_errors : bool
            boolean to assess whether to color according to the error in class prediction; if show_errors=False, then either the ground truth or the predicted class labels will be used to color the datapoints; otherwise, the points will be restricted to the set of known matchings (even if withzeros=True) and colored according to the identity between the ground truth and the predicted labels for each (user, item) pair
        verbose : bool
            prints out information at each step
        '''
        assert fontsize > 0
        assert random_state >= 0
        nvalues = self.folds.data.shape[0]
        assert (X is None and y is None) or ((X.shape[0]==y.shape[0]==nvalues))
        assert predictions is None or (predictions.data.shape[0]==nvalues)
        if (self.users.shape[0]==0 or self.items.shape[0]==0):
            if (verbose):
                print("<datasets.visualize> Can't plot (no item/user feature matrix).")
            return None
        assert predictions is None or (((predictions.toarray()==1)|(predictions.toarray()==-1)|(predictions.toarray()==0)).all())
        if (X is None and y is None):
            if (verbose):
                print("<datasets.visualize> Imputation of missing values by average row-value, standard scaling")
            subselect_size = max(2,min(int(5e7)//nvalues+1, nvalues))
            subselect_size = min(subselect_size, min(self.users.shape[0],self.items.shape[0]))
            ## Preprocessed (item, user) pair feature matrix and corresponding outcome vector
            X, y, _, _ = meanimputation_standardize(self, subset=subselect_size, inf=2, verbose=verbose)
            use_inputX=False
        else:
            predictions = None
            show_errors = False
            use_inputX=True
        markers = np.column_stack((self.folds.row, self.folds.col))
        ## True (known and unknown) ratings for all items and users in the dataset
        ## item i, user u, rating r
        markers = np.concatenate((markers, y.reshape(-1,1)), axis=1)
        if ((use_ratings) and (not use_inputX)):
            predictions = markers
        ## 1. predictions=None: Plots datapoints according to ground truth annotations
        if (predictions is None):
            show_errors = False
            ## item i, user u, rating r, scatter style (color + marker shape)
            all_pairs = np.array([[{-1:"r.", 1:"g.", 0:"y."}[k]] for i,j,k in markers.tolist()])
            all_pairs = np.concatenate((markers, all_pairs), axis=1)
            assert all_pairs.shape[1]==4 and all_pairs.shape[0]==nvalues
        else:
            ## 2. predictions!=None: Plots datapoints according to predicted annotations
            if (not use_ratings):
                classes = dict(zip([(i,j) for i,j in zip(self.folds.row, self.folds.col)], predictions.data))
                all_pairs = np.array([[classes.get((i,j), 0)] for i, j in markers[:,:2].astype(int).tolist()])
                all_pairs = np.concatenate((markers, all_pairs), axis=1)
            else:
                all_pairs = np.concatenate((markers, markers[:,2]), axis=1)
            ## 2.a. predictions!=None: Only for datapoints with known ratings
            if (show_errors):
                values = np.array([[{0:"r", 1:"g", -1:"y"}[int(true==pred)-int(true==0)]+{-1:"v", 1:"+", 0:"."}[true]] for [i,j,true,pred] in all_pairs.tolist()])
                ## Predicted ratings for all known pairs of items and users in the dataset
                ## item i, user u, rating r, scatter style (color + marker shape)
                all_pairs = np.array(np.concatenate((all_pairs[:,[0,1,3]], values), axis=1), dtype=object)
                all_pairs = all_pairs[y!=0,:]
                X = X[y!=0,:]
                y = y[y!=0]
                assert all_pairs.shape[1]==4
            ## 2.b. predictions!=None: For all datapoints
            else:
                values = np.array([[{-1:"r", 1:"g", 0:"y"}[pred]+{-1:"v", 1:"+", 0:"."}[true]] for [i,j,true,pred] in all_pairs.tolist()])
                ## Predicted ratings for all known pairs of items and users in the dataset
                ## item i, user u, rating r, scatter style (color + marker shape)
                all_pairs = np.array(np.concatenate((all_pairs[:,[0,1,3]], values), axis=1), dtype=object)
                assert all_pairs.shape[1]==4 and all_pairs.shape[0]==X.shape[0]
        all_pairs[:,:-1] = all_pairs[:,:-1].astype(float).astype(int)
        if (verbose):
            print("<datasets.visualize> Reducing dimension and plotting matrix X of size %d x %d" % X.shape)
        dimred_args.update({"n_components":min(2,X.shape[1]), "random_state":random_state})
        use_pca = ("n_neighbors" not in dimred_args)
        if (use_pca):
            with np.errstate(invalid="ignore"): # for NaN or 0 variance matrices
                pca = PCA(**dimred_args)
                dimred_X = pca.fit_transform(X)
                var12 = pca.explained_variance_ratio_[:2]*100
        else:
            dimred_args.update({"n_neighbors":max(5,min(dimred_args["n_neighbors"],min(50,all_pairs.shape[0])))})
            dimred_args.update({"min_dist":max(0.5,min(dimred_args.get("min_dist", 0.1), 0.001))})
            dimred_args.update({"metric":dimred_args.get("metric", 'correlation')})
            if (verbose):
                print("<datasets.visualize> n_neighbors = %d\tmin_dist = %.2f\tmetric = %s" % (dimred_args["n_neighbors"], dimred_args["min_dist"], dimred_args["metric"]))
            with np.errstate(invalid="ignore"): # for NaN or 0 variance matrices
                umap_model = umap.UMAP(**dimred_args)
                dimred_X = umap_model.fit_transform(X, y)
                var12 = [np.nan]*2
        ## Put points in the front layer
        layer = {"g.": 1, "r.": 1, "y.": 0} if (predictions is None) else ({"g.": 0, "r.": 0, "y.": 0, "g+": 0, "r+": 1, "y+": 0, "gv": 0, "rv": 0, "yv": 0} if (not show_errors) else {"g.": 0, "r.": 0, "y.": 0, "g+": 0, "r+": 1, "y+": 0, "gv": 1, "rv": 0, "yv": 0})
        ## More visible points
        alpha = {"g.": 0.75, "r.": 1, "y.": 0.1}
        plt.figure(figsize=figsize)
        if (X.shape[1]>1):
            ## Prints a PCA / UMAP
            for mkr in np.unique(np.ravel(all_pairs[:,3])).tolist():
                all_pairs_k = np.argwhere(all_pairs[:,3]==mkr)[:,0].tolist()
                if ((not withzeros) and (((predictions is None) and (mkr=="y.")) or ((predictions is not None) and (mkr[-1]==".")))):
                    plt.scatter(dimred_X[all_pairs_k,0], dimred_X[all_pairs_k,1], c="w", marker=".", zorder=0, alpha=0)
                else:
                    plt.scatter(dimred_X[all_pairs_k,0], dimred_X[all_pairs_k,1], c=mkr[0], marker=mkr[1], zorder=layer[mkr], alpha=alpha[mkr] if (predictions is None) else (0.8 if (not show_errors) else 1))
            if (show_errors):
                handles = [mlines.Line2D([], [], color={'r':'red','g':'green','y':'yellow'}[k[0]], 
                    label={'r':'error   ','g':'correct', 'y': 'unknown'}[k[0]]+" "+({".": "[ 0]", "+": "[ 1]", "v": "[-1]"}[k[1]]),
		    marker=k[1] if (predictions is not None) else '.', markersize=fontsize,
		    ) for k in np.unique(np.asarray(all_pairs[:,-1], dtype=str)).tolist() if (withzeros or k[0]!="y")]
            else:
                if (predictions is None):
                    handles = [mlines.Line2D([], [], color={'r':'red','g':'green','y':'yellow'}[k[0]], 
                        label={'r':'[-1]','y':'[ 0]','g':"[ 1]"}[k[0]],
		        marker=k[1] if (predictions is not None) else '.', markersize=fontsize,
		        ) for k in np.unique(np.asarray(all_pairs[:,-1], dtype=str)).tolist() if (withzeros or k[0]!="y")]
                else:
                    handles = [mlines.Line2D([], [], color={'r':'red','g':'green','y':'yellow'}[k[0]], 
                        label={'r':'neg','g':"pos", 'y':'unl'}[k[0]]+" "+{".": "[ 0]", "+": "[ 1]", "v": "[-1]"}[k[1]],
		        marker=k[1] if (predictions is not None) else '.', markersize=fontsize,
		        ) for k in np.unique(np.asarray(all_pairs[:,-1], dtype=str)).tolist() if (withzeros or k[0]!="y")]
            plt.xticks(fontsize=fontsize, rotation=90)
            plt.yticks(fontsize=fontsize)
            plt.ylabel(("PC2 ("+str(int(var12[1]))+"%)" if (use_pca) else "Dim2") if (not np.isnan(var12[1])) else "C2", fontsize=fontsize)
            plt.xlabel((("PC1 ("+str(int(var12[0]))+"%)" if (use_pca) else "Dim1")) if (not np.isnan(var12[0])) else "C1", fontsize=fontsize)
            plt.title("on %d features" % X.shape[1], fontsize=fontsize//2)
            plt.legend(handles=handles, fontsize=fontsize, loc='upper right', bbox_to_anchor=(1.6,0.9))
            plt.show()
        elif ((dimred_X!=0).any()):
            ## Prints a heatmap according to values in X
            X_reshape_folds = coo_array((X.ravel(), (self.folds.row, self.folds.col)), 
                            shape=self.folds.shape)
            X_heatmap = X_reshape_folds.toarray()
            annot = self.ratings.toarray().astype(int).astype(str)
            annot[annot=="0"] = "" #unknown
            annot[annot=="1"] = "+"  #positive
            annot[annot=="-1"] = "*"  #negative
            keep_ids_y = np.abs(X_heatmap).sum(axis=1)>0 
            keep_ids_x = np.abs(X_heatmap).sum(axis=0)>0 
            X_heatmap = X_heatmap[keep_ids_y,:][:,keep_ids_x]
            annot = annot[keep_ids_y,:][:,keep_ids_x]
            h = sns.clustermap(X_heatmap, method='average', cmap="viridis", metric=metric, annot=annot, fmt="s", figsize=figsize)
            h.ax_heatmap.set_xlabel("Disease", fontsize=fontsize) 
            h.ax_heatmap.set_ylabel("Drug", fontsize=fontsize)
            h.ax_heatmap.set_xticklabels([])
            h.ax_heatmap.set_yticklabels([]) 
            handles = [mlines.Line2D([], [], color="black", marker=k) for k in ["+", "*"] if (k in annot)]
            h.ax_heatmap.legend(handles, ["Positive", "Negative"], fontsize=fontsize, loc='upper right', bbox_to_anchor=(1.6,0.9))
            plt.show()
        else:
            if (verbose):
                print("<stanscofi.dataset.visualize> Matrix is empty, can't plot!")
        return None

    def subset(self, folds, subset_name="subset"):
        '''
        Obtains a subset of a stanscofi.Dataset based on a set of user and item indices

        ...

        Parameters
        ----------
        folds : COO-array of shape (n_items, n_users)
            an array which contains values in {0, 1} describing the unavailable and available user-item matchings in ratings
        subset_name : str
            name of the newly created stanscofi.Dataset

        Returns
        ----------
        subset : stanscofi.Dataset
            dataset corresponding to the folds in input
        '''
        if (np.prod(folds.shape)==0):
            raise ValueError("Fold is empty!")
        assert folds.shape==self.folds.shape
        #data = self.ratings.toarray()[folds.row, folds.col].ravel()
        sfolds = np.asarray(folds.toarray(), dtype=float)
        sfolds[sfolds==0] = np.nan
        ratings = self.ratings.toarray() * sfolds
        ratings = pd.DataFrame(ratings, index=self.item_list, columns=self.user_list)
        users = pd.DataFrame(self.users.toarray(), index=self.user_features, columns=self.user_list)
        items = pd.DataFrame(self.items.toarray(), index=self.item_features, columns=self.item_list)
        subset =  Dataset(ratings=ratings, users=users, items=items, same_item_user_features=self.same_item_user_features, name=subset_name)
        return subset