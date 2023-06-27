#coding: utf-8

import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from sklearn.decomposition import PCA
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*he 'nopython' keyword argument was not supplied to the 'numba.jit' decorator.*")
    import umap
from sklearn.preprocessing import StandardScaler

import stanscofi.utils
import stanscofi.preprocessing

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
    ratings_mat : array-like of shape (n_items, n_users)
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
    ratings = np.zeros((nitems*nusers, 3))
    ids = np.argwhere(np.ones((nitems, nusers)))
    ratings[:,0] = ids[:,0]
    ratings[:,1] = ids[:,1]
    positive = (ratings[:,0]<npositive)&(ratings[:,1]<npositive)
    negative_u = (ratings[:,0]>=npositive)&(ratings[:,0]<nnegative+npositive+1)
    negative_i = (ratings[:,1]>=npositive)&(ratings[:,1]<nnegative+npositive+1)
    ratings[positive,2] = 1
    ratings[negative_u&negative_i,2] = -1
    ratings = pd.DataFrame(ratings, columns=["user","item","rating"], index=range(ratings.shape[0]))
    ## Input to stanscofi
    ratings_mat = stanscofi.utils.ratings2matrix(ratings, user_col="user", item_col="item", rating_col="rating")
    users = pd.DataFrame(users, index=range(nfeatures//2), columns=range(nusers))
    items = pd.DataFrame(items, index=range(nfeatures//2), columns=range(nitems))
    return {"ratings_mat": ratings_mat, "users": users, "items": items}

class Dataset(object):
    '''
    A class used to encode a drug repurposing dataset

    ...

    Parameters
    ----------
    ratings_mat : array-like of shape (n_items, n_users)
        a matrix which contains values in {-1, 0, 1} describing the known and unknown user-item matchings
    users : array-like of shape (n_item_features, n_items)
        a list of the item feature names in the order of column indices in ratings_mat
    items : array-like of shape (n_user_features, n_users)
        a list of the item feature names in the order of column indices in ratings_mat
    same_item_user_features : bool
        a list of the item feature names in the order of column indices in ratings_mat
    name : str
        the name of the dataset (if it exists)

    Attributes
    ----------
    name : str
        the name of the dataset (if it exists)
    ratings_mat : array-like of shape (n_items, n_users)
        a matrix which contains values in {-1, 0, 1} describing the known and unknown user-item matchings
    users : array-like of shape (n_item_features, n_items)
        a list of the item feature names in the order of column indices in ratings_mat
    items : array-like of shape (n_user_features, n_users)
        a list of the item feature names in the order of column indices in ratings_mat
    item_list : list of str
        a list of the item names in the order of row indices in ratings_mat
    user_list : list of str
        a list of the item names in the order of column indices in ratings_mat
    item_features : list of str
        a list of the item feature names in the order of column indices in ratings_mat
    user_features : list of str
        a list of the item feature names in the order of column indices in ratings_mat
    ratings : array-like of shape (n_ratings, 3)
        a list of the item feature names in the order of column indices in ratings_mat
    same_item_user_features : bool
        a list of the item feature names in the order of column indices in ratings_mat

    Methods
    -------
    __init__(ratings_mat=None, users=None, items=None, same_item_user_features=False, name="dataset")
        Initialize the Dataset object and creates all attributes
    summary(sep="-"*70)
        Prints out the characteristics of the drug repurposing dataset
    visualize(withzeros=False, X=None, y=None, figsize=(5,5), fontsize=20, dimred_args={}, predictions=None, use_ratings=False, random_state=1234, show_errors=False, verbose=False)
        Plots datapoints in the dataset annotated by the ground truth or predicted ratings
    get_folds(folds, subset_name="subset")
        Creates a subset of the dataset based on the folds given as input
    '''
    def __init__(self, ratings_mat=None, users=None, items=None, same_item_user_features=False, name="dataset"):
        '''
        Creates an instance of stanscofi.Dataset

        ...

        Parameters
        ----------
        ratings_mat : array-like of shape (n_items, n_users)
            a matrix which contains values in {-1, 0, 1} describing the known and unknown user-item matchings
        users : array-like of shape (n_item_features, n_items)
            a list of the item feature names in the order of column indices in ratings_mat
        items : array-like of shape (n_user_features, n_users)
            a list of the item feature names in the order of column indices in ratings_mat
        same_item_user_features : bool
            a list of the item feature names in the order of column indices in ratings_mat
        name : str
            the name of the dataset (if it exists)
        '''
        assert ratings_mat is not None
        assert users is not None
        assert items is not None
        self.item_list = list(ratings_mat.index)
        self.user_list = list(ratings_mat.columns)
        users = users[self.user_list]
        items = items[self.item_list]
        if (same_item_user_features):
            features = list(set(list(users.index)).intersection(set(list(items.index))))
            assert len(features)>0
            self.user_features = features
            self.item_features = features
            users = users.loc[features]
            items = items.loc[features]
        else:
            self.user_features = list(users.index)
            self.item_features = list(items.index)
        self.ratings_mat = ratings_mat.values
        self.ratings = stanscofi.utils.matrix2ratings(ratings_mat, user_col="ind_id", item_col="drug_id", rating_col="rating").values
        assert all([a in users.columns for a in self.ratings[:,0]])
        assert all([a in items.columns for a in self.ratings[:,1]])
        self.ratings[:,0] = [self.user_list.index(x) for x in self.ratings[:,0]] 
        self.ratings[:,1] = [self.item_list.index(x) for x in self.ratings[:,1]] 
        self.users = users.values
        self.items = items.values
        self.name = name
        self.same_item_user_features = same_item_user_features

    def summary(self, sep="-"*70):
        '''
        Prints out a summary of the contents of a stanscofi.Dataset: the number of items, users, the number of positive, negative, unknown matchings, the sparsity number, and the shape and percentage of missing values in the item and user feature matrices

        ...

        Parameters
        ----------
        sep : str
            separator for pretty printing
        '''
        A = pd.DataFrame(self.ratings_mat, index=self.item_list, columns=self.user_list)
        print(sep)
        print("* Matching matrix:")
        ratings = pd.DataFrame(self.ratings, index=range(self.ratings.shape[0]), columns=["user", "item", "rating"])
        ratings2 = ratings.copy()
        ratings2["item"] = ratings2["item"].astype(str)
        args = [len(np.unique(ratings2["item"])), len(np.unique(ratings2["user"]))]
        args += [ratings2.loc[ratings2["rating"]==v].shape[0] for v in [1,-1]]
        args += [np.prod(A.shape)-args[2]-args[3],(args[2]+args[3])*100/np.prod(A.shape)]
        dataset_str = "Ratings: %d drugs\t%d diseases involved in at least one known matching\n%d positive, %d negative, %d unknown matchings\nSparsity for drugs/diseases involved in at least one known matching: %.2f perc."
        print(dataset_str % tuple(args))
        print(sep[:len(sep)//2])
        print("* Feature matrices:")
        if (self.items.shape[0]>0):
            print("Total #Drugs: %d\t#Drug features: %d\tPerc. Missing features: %d" % (self.items.shape[1], self.items.shape[0], np.sum(np.isnan(self.items))*100/(self.items.shape[0]*self.items.shape[1])))
        if (self.users.shape[0]>0):
            print("Total #Diseases: %d\t#Disease features: %d\tPerc. Missing features: %d" % (self.users.shape[1], self.users.shape[0], np.sum(np.isnan(self.users))*100/(self.users.shape[0]*self.users.shape[1])))
        if (self.users.shape[0]+self.items.shape[0]==0):
            print("No feature matrices.")
        print(sep+"\n")

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
        nvalues = np.prod(self.ratings_mat.shape)
        assert (X is None and y is None) or ((X.shape[0]==y.shape[0]==nvalues))
        assert predictions is None or ((predictions.shape[1]==3) and (predictions.shape[0]==nvalues))
        if (self.users.shape[0]==0 or self.items.shape[0]==0):
            if (verbose):
                print("<datasets.visualize> Can't plot (no item/user feature matrix).")
            return None
        assert predictions is None or (((predictions[:,-1]==1)|(predictions[:,-1]==-1)).all())
        if (X is None and y is None):
            if (verbose):
                print("<datasets.visualize> Imputation of missing values by average row-value, standard scaling")
            subselect_size = max(2,min(int(5e7)//nvalues+1, nvalues))
            ## Preprocessed (item, user) pair feature matrix and corresponding outcome vector
            X, y, _, _ = stanscofi.preprocessing.meanimputation_standardize(self, subset=min(subselect_size, min(self.users.shape[0],self.items.shape[0])), inf=2, verbose=verbose)
            use_inputX=False
        else:
            predictions = None
            show_errors = False
            use_inputX=True
        markers = np.argwhere(np.ones(self.ratings_mat.shape))
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
            predictions = predictions.astype(int)
            ## 2. predictions!=None: Plots datapoints according to predicted annotations
            if (not use_ratings):
                classes = dict(zip([(i,j) for i,j in predictions[:,:2].tolist()], predictions[:,2].flatten()))
                all_pairs = np.array([[classes[(j,i)]] for i, j in markers[:,:2].tolist()])
                all_pairs = np.concatenate((markers, all_pairs), axis=1)
            else:
                all_pairs = np.concatenate((markers, markers[:,2]), axis=1)
            ## 2.a. predictions!=None: Only for datapoints with known ratings
            if (show_errors):
                values = np.array([[{0:"r", 1:"g", -1:"y"}[int(true==pred)-int(true==0)]+{-1:"v", 1:"+", 0:"."}[true]] for [i,j,true,pred] in all_pairs.tolist()])
                ## Predicted ratings for all known pairs of items and users in the dataset
                ## item i, user u, rating r, scatter style (color + marker shape)
                all_pairs = np.concatenate((all_pairs[:,[0,1,3]],values), axis=1, dtype=object)
                all_pairs = all_pairs[y!=0,:]
                X = X[y!=0,:]
                assert all_pairs.shape[1]==4
            ## 2.b. predictions!=None: For all datapoints
            else:
                values = np.array([[{-1:"r", 1:"g", 0:"y"}[pred]+{-1:"v", 1:"+", 0:"."}[true]] for [i,j,true,pred] in all_pairs.tolist()])
                ## Predicted ratings for all known pairs of items and users in the dataset
                ## item i, user u, rating r, scatter style (color + marker shape)
                all_pairs = np.concatenate((all_pairs[:,[0,1,3]],values), axis=1, dtype=object)
                assert all_pairs.shape[1]==4 and all_pairs.shape[0]==X.shape[0]
        all_pairs[:,:-1] = all_pairs[:,:-1].astype(int)
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
            X_heatmap = X.reshape(self.ratings_mat.shape)
            annot = self.ratings_mat.copy().astype(str)
            annot[annot=="0"] = "" #unknown
            annot[annot=="1"] = "+"  #positive
            annot[annot=="-1"] = "*"  #negative
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

    def get_folds(self, folds, subset_name="subset"):
        '''
        Obtains a subset of a stanscofi.Dataset based on a set of user and item indices

        ...

        Parameters
        ----------
        folds : array-like of shape (n_ratings, 3)
            a matrix which contains the user indices (column 1), the item indices (column 2) and the class for the corresponding (user, item) pair (value in {-1, 0, 1} in column 3)
        subset_name : str
            name of the newly created stanscofi.Dataset

        Returns
        subset : stanscofi.Dataset
            dataset corresponding to the folds in input
        ----------
        '''
        if (len(folds)==0):
            raise ValueError("Fold is empty!")
        assert folds.shape[1]==3
        assert np.max(folds[:,0])<=self.ratings_mat.shape[1]
        assert np.max(folds[:,1])<=self.ratings_mat.shape[0]
        assert ((folds[:,-1]==1)|(folds[:,-1]==-1)|(folds[:,-1]==0)).all()
        ratings = pd.DataFrame(folds, index=range(folds.shape[0]), columns=["user","item","rating"])
        ratings["user"] = [self.user_list[x] for x in ratings["user"]]
        ratings["item"] = [self.item_list[x] for x in ratings["item"]]
        user_lst = np.unique(folds[:,0]).tolist() ## order of unique
        item_lst = np.unique(folds[:,1]).tolist() ## order of unique
        A = stanscofi.utils.ratings2matrix(ratings, user_col="user", item_col="item", rating_col="rating")
        P = pd.DataFrame(self.users[:,user_lst], index=self.user_features, 
                columns=[self.user_list[i] for i in user_lst])
        S = pd.DataFrame(self.items[:,item_lst], index=self.item_features, 
                columns=[self.item_list[i] for i in item_lst])
        return Dataset(A, P, S, name=subset_name, same_item_user_features=self.same_item_user_features)