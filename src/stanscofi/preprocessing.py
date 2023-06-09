#coding:utf-8

import numpy as np

from joblib import Parallel, delayed
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class CustomScaler(object):
    '''
    A class used to encode a simple preprocessing pipeline for feature matrices. Does mean imputation for features, feature filtering, correction of infinity errors and standardization

    ...

    Parameters
    ----------
    posinf : int
        Value to replace infinity (positive) values
    neginf : int
        Value to replace infinity (negative) values

    Attributes
    ----------
    imputer : None or sklearn.impute.SimpleImputer instance
        Class for imputation of values
    scaler : None or sklearn.preprocessing.StandardScaler
        Class for standardization of values
    filter : None or list
        List of selected features (Top-N in terms of variance)

    Methods
    -------
    __init__(params)
        Initialize the scaler (with unfitted attributes)
    fit_transform(mat, subset=None, verbose=False)
        Fits classes and transforms a matrix
    '''
    def __init__(self, posinf, neginf):
        '''
        Initialize the scaler (with unfitted imputer, standardscaler and filter attributes)
        '''
        self.imputer = None
        self.scaler = None
        self.remove_nan = []
        self.filter = None
        self.posinf = None
        self.neginf = None

    def fit_transform(self, mat, subset=None, verbose=False): ## elements x features
        '''
        Fits each attribute of the scaler and transform a feature matrix. Does mean imputation for features, feature filtering, correction of infinity errors and standardization

        ...

        Parameters
        ----------
        mat : array-like of shape (n_samples, n_features)
            matrix which should be preprocessed
        subset : None or int
            number of features to keep in feature matrix (Top-N in variance); if it is None, attribute filter is either initialized (if it is equal to None) or used to filter features
        verbose : bool
            prints out information

        Returns
        -------
        mat_nan : array-like of shape (n_samples, n_features)
            Preprocessed matrix
        '''
        mat_nan = np.nan_to_num(mat, copy=True, nan=np.nan, posinf=self.posinf, neginf=self.neginf)
        assert mat_nan.shape==mat.shape
        if ((subset is not None) or (self.filter is not None)):
            if (verbose):
                print("<preprocessing.CustomScaler> Selecting the %d most variable features out of %d..." % (subset, mat_nan.shape[1]))
            if ((subset is not None) and (self.filter is None)):
                with np.errstate(over="ignore"):
                    x_vars = [np.nanvar(mat_nan[:,i]) if (np.sum(~np.isnan(mat_nan[:,i]))>0) else 0 for i in range(mat_nan.shape[1])]
                    x_vars = [x if (not np.isnan(x) and not np.isinf(x)) else 0 for x in x_vars]
                    x_ids_vars = np.argsort(x_vars).tolist()
                    features = x_ids_vars[-subset:]
                    self.filter = features
            mat_nan = mat_nan[:,self.filter]
        assert mat_nan.shape[1]==(mat.shape[1] if (self.filter is None) else len(self.filter)) and mat_nan.shape[0]==mat.shape[0]
        mat_nan[:,np.sum(~np.isnan(mat_nan), axis=0)==0] = 0 # avoid overflow warning from SimpleImputer
        if (verbose):
            print("<preprocessing.CustomScaler> %d perc. of missing values (||.|| = %f)..." % (100*np.sum(np.isnan(mat_nan))/np.prod(list(mat_nan.shape)), np.linalg.norm(mat_nan)), end=" ")
        if (self.imputer is None and np.sum(np.isnan(mat_nan))>0):
            with np.errstate(under="ignore"):
                self.imputer = SimpleImputer(missing_values=np.nan, strategy='mean', keep_empty_features=True)
                mat_nan = self.imputer.fit_transform(mat_nan)
        else:
            if (np.sum(np.isnan(mat_nan))>0):
                mat_nan = self.imputer.transform(mat_nan)
        if (verbose):
            print("Final ||.|| = %f" % (np.linalg.norm(mat_nan)))
        assert mat_nan.shape[0]==mat.shape[0]
        if (self.scaler is None):
            self.scaler = StandardScaler()
            mat_nan_std = self.scaler.fit_transform(mat_nan)
        else:
            mat_nan_std = self.scaler.transform(mat_nan)
        assert mat_nan_std.shape[1]==(mat.shape[1] if (self.filter is None) else len(self.filter)) and mat_nan_std.shape[0]==mat.shape[0]
        return mat_nan_std

def meanimputation_standardize(dataset, subset=None, scalerS=None, scalerP=None, inf=int(1e1), verbose=False):
    '''
    Computes a single feature matrix and response vector from a drug repurposing dataset, by imputation by the average value of a feature for missing values and by centering and standardizing user and item feature matrices and concatenating them

    ...

    Parameters
    ----------
    dataset : stanscofi.Dataset
        dataset which should be transformed, with n_items items (with n_item_features features) and n_users users (with n_user_features features) where missing values are denoted by numpy.nan
    subset : None or int
        number of features to keep in item feature matrix, and in user feature matrix (selecting the ones with highest variance)
    scalerS : None or sklearn.preprocessing.StandardScaler instance
        scaler for items
    scalerP : None or sklearn.preprocessing.StandardScaler instance
        scaler for users
    verbose : bool
        prints out information

    Returns
    ----------
    X : array-like of shape (n_items x n_users, n_item_features+n_user_features)
        the feature matrix
    y : array-like of shape (n_items x n_users, )
        the response/outcome vector
    scalerS : None or stanscofi.models.CustomScaler instance
        scaler for items; if the input value was None, returns the scaler fitted on item feature vectors
    scalerP : None or stanscofi.models.CustomScaler instance
        scaler for users; if the input value was None, returns the scaler fitted on user feature vectors
    '''
    y = np.ravel(dataset.ratings_mat.flatten())
    if (scalerS is None):
        scalerS = CustomScaler(posinf=inf, neginf=-inf)
    if (verbose):
        print("<preprocessing.meanimputation_standardize> Preprocessing of item feature matrix...")
    S_ = scalerS.fit_transform(dataset.items.T.copy(), subset=subset, verbose=verbose) ## items x features=subset
    if (scalerP is None):
        scalerP = CustomScaler(posinf=inf, neginf=-inf)
    if (verbose):
        print("<preprocessing.meanimputation_standardize> Preprocessing of user feature matrix...")
    P_ = scalerP.fit_transform(dataset.users.T.copy(), subset=subset, verbose=verbose) ## users x features=subset
    ids = np.argwhere(np.ones(dataset.ratings_mat.shape)) 
    SS = np.concatenate(tuple([S_[i,:].reshape(-1,1) for i, _ in ids.tolist()]), axis=1) ## (item, user) pairs x features
    PP = np.concatenate(tuple([P_[j,:].reshape(-1,1) for _, j in ids.tolist()]), axis=1) ## (item, user) pairs x features
    X = np.concatenate((SS, PP), axis=0).T ## (item, user) pairs x item/user features
    return X, y, scalerS, scalerP

def same_feature_preprocessing(dataset):
    '''
    If the users and items have the same features in the dataset, then a simple way to combine the user and item feature matrices is to multiply the feature vectors coefficient per coefficient.

    ...

    Parameters
    ----------
    dataset : stanscofi.Dataset
        dataset which should be transformed, with n_items items (with n_item_features features) and n_users users (with n_user_features features) where n_item_features==n_user_features and dataset.same_item_user_features==True, and missing values are denoted by 0 or numpy.nan

    Returns
    ----------
    X : array-like of shape (n_items x n_users, n_item_features+n_user_features)
        the feature matrix
    y : array-like of shape (n_items x n_users, )
        the response/outcome vector
    '''
    assert dataset.same_item_user_features==True
    y = np.ravel(dataset.ratings_mat.flatten())
    S_, P_ = dataset.items.T.copy(), dataset.users.T.copy()
    P_[np.isnan(P_)] = 0
    S_[np.isnan(S_)] = 0
    ids = np.argwhere(np.ones(dataset.ratings_mat.shape)) # (item, user) pairs
    X = np.concatenate(tuple([np.multiply(S_[i, :], P_[j, :]).reshape(-1,1) for i, j in ids.tolist()]), axis=1).T ## pairs x features
    return X, y

## https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
def cartesian_product_transpose(*arrays):
    broadcastable = np.ix_(*arrays)
    broadcasted = np.broadcast_arrays(*broadcastable)
    rows, cols = np.prod(broadcasted[0].shape), len(broadcasted)
    dtype = np.result_type(*arrays)
    out = np.empty(rows * cols, dtype=dtype)
    start, end = 0, rows
    for a in broadcasted:
        out[start:end] = a.reshape(-1)
        start, end = end, end + rows
    return out.reshape(cols, rows).T

def Perlman_procedure(dataset, njobs=1, sep_feature="-", missing=-666, inf=2, verbose=False):
    '''
    Method for combining (several) item and user similarity matrices (reference DOI: 10.1089/cmb.2010.0213). Instead of concatenating item and user features for a given pair, resulting in a vector of size (n_items x n_item_matrices)+(n_users x n_user_matrices), compute a single score per pair of (item_matrix, user_matrix) for each (item, user) pair, resulting in a vector of size (n_item_matrices) x (n_user_matrices).

    The score for any item i, user u, item-item similarity fi and user-user similarity fu is
    score_{fi,fu}(i,u) = max { sqrt(fi(dr, dr') x fu(di', di))  | (i',u')!=(i,u), fi(dr, dr')!=NaN, fu(di', di)!=NaN, rating(i',u')!=0 }

    Then the final feature matrix is X = (X_{i,j})_{i,j} for i a (item, user) pair and j a (item similarity, user similarity) pair

    ...

    Parameters
    ----------
    dataset : stanscofi.Dataset
        dataset which should be transformed, with n_items items (with n_item_features features) and n_users users (with n_user_features features) with the following attributes
            ratings_mat : array-like of shape (n_ratings, 3)
                a matrix which contains the user indices (column 1), the item indices (column 2) and the class for the corresponding (user, item) pair (value in {-1, 0, 1} in column 3), where 0 stands for unknown class, -1 for negative and 1 for positive classes
            users : array-like of shape (n_diseases x n_disease_features, n_diseases)
                concatenation of n_disease_features drug similarity matrices of shape (n_diseases, n_diseases), where values in user_features are denoted by "<feature><sep_feature><disease>" and missing values are denoted by numpy.nan; if the prefix in "<feature><sep_feature>" is missing, it is assumed that users is a single similarity matrix (n_user_matrices=1)
            items : array-like of shape (n_drugs x n_drug_features, n_drugs)
                concatenation of n_drug_features drug similarity matrices of shape (n_drugs, n_drugs), where values in item_features are denoted by "<feature><sep_feature><drug>" and missing values are denoted by numpy.nan; if the prefix in "<feature><sep_feature>" is missing, it is assumed that items is a single similarity matrix (n_item_matrices=1)
        NaN values are replaced by 0, whereas infinite values are replaced by inf (parameter below).
    njobs : int
        number of jobs to run in parallel
    sep_feature : str
        separator between feature type and element in the feature matrices in dataset
    missing : int
        placeholder value that should be different from any feature name
    inf : int
        Value that replaces infinite values in the dataset (inf for +infinity, -inf for -infinity)
    verbose : bool
        prints out information

    Returns
    ----------
    X : array-like of shape (n_items x n_users, n_item_features x n_user_features)
        the feature matrix
    y : array-like of shape (n_items x n_users, )
        the response/outcome vector 
    '''
    assert njobs > 0
    is_user = sum([int(sep_feature in str(x)) for x in dataset.user_features])
    is_item = sum([int(sep_feature in str(x)) for x in dataset.item_features])
    assert is_user in [0, dataset.users.shape[0]]
    assert is_item in [0, dataset.items.shape[0]]
    A, P, S = dataset.ratings_mat.copy(), dataset.users.copy(), dataset.items.copy()
    np.nan_to_num(P, copy=False, nan=0, posinf=inf, neginf=-inf)
    np.nan_to_num(S, copy=False, nan=0, posinf=inf, neginf=-inf)
    ## Ensure positive values in P and S
    P = P-np.tile(np.nanmin(P, axis=1), (P.shape[1], 1)).T
    S = S-np.tile(np.nanmin(S, axis=1), (S.shape[1], 1)).T
    y = np.ravel(A.flatten())
    ## All item,user pairs (items x users)
    ids = np.argwhere(np.ones(A.shape)) 
    ## All types of features (length=#features)
    nuser_features = np.array([missing]*len(dataset.user_features) if (is_user==0) else [x.split(sep_feature)[0] for x in dataset.user_features])
    nitem_features = np.array([missing]*len(dataset.item_features) if (is_item==0) else [x.split(sep_feature)[0] for x in dataset.item_features])
    ## All item/user corresponding to each type of feature (length=#features)
    ## In subset datasets some features might relate to absent diseases/drugs
    nuser_nms = np.array([dataset.user_list.index(x) if (x in dataset.user_list) else missing for x in (dataset.user_features if (is_user==0) else [y.split(sep_feature)[1] for y in dataset.user_features])])
    nitem_nms = np.array([dataset.item_list.index(x) if (x in dataset.item_list) else missing for x in (dataset.item_features if (is_item==0) else [y.split(sep_feature)[1] for y in dataset.item_features])])
    Nuf, Nif = len(set(nuser_features)), len(set(nitem_features))
    if (verbose):
        print("<preprocessing.Perlman_procedure> %d item similarities, %d user similarities" % (Nuf, Nif))
    X = np.zeros((ids.shape[0], Nuf*Nif)) ## (item, user) pairs x feature type
    if (verbose):
        print("<preprocessing.Perlman_procedure> For %d ratings pairs, identified %d features" % X.shape)
    item_feature_mask = {fi: (nitem_features==fi).astype(int) for fi in set(nitem_features)}
    item_mask = [S[:,i] * (nitem_nms!=i).astype(int) for i in range(A.shape[0])]
    user_feature_mask = {fu: (nuser_features==fu).astype(int) for fu in set(nuser_features)}
    user_mask = [P[:,u] * (nuser_nms!=u).astype(int) for u in range(A.shape[1])]
    known = np.array([0 if ((nitem_nms[i]==missing) or (nuser_nms[u]==missing)) else int(A[nitem_nms[i],nuser_nms[u]]!=0) for i in range(S.shape[0]) for u in range(P.shape[0])])
    features = [[a,b] for a in set(nitem_features) for b in set(nuser_features)]
    def single_run(fi, fu, known, item_feature_mask, item_mask, user_feature_mask, user_mask, ids):
        if (verbose):
            print("<preprocessing.Perlman_procedure> Considering item feature '%s' and user feature '%s' (%d jobs)" % (str(fi) if (fi!=missing) else "{1}", str(fu) if (fu!=missing) else "{1}", njobs))
        vals = [None]*ids.shape[0]
        for ii, [i, u] in enumerate(ids.tolist()):
            item_mat = item_feature_mask[fi] * item_mask[i]
            user_mat = user_feature_mask[fu] * user_mask[u]
            all_sim = np.prod(cartesian_product_transpose(item_mat, user_mat), axis=1)
            vals[ii] = np.max(np.multiply(all_sim, known))
        return vals
    if (njobs==1):
        vals = [single_run(fi, fu, known, item_feature_mask, item_mask, user_feature_mask, user_mask, ids) for [fi, fu] in tqdm(features)]
    else:
        vals = Parallel(n_jobs=min(njobs, len(features)), backend='loky')(delayed(single_run)(fi, fu, known, item_feature_mask, item_mask, user_feature_mask, user_mask, ids) for [fi, fu] in tqdm(features))
    X = np.array(vals).T
    return np.sqrt(X), y