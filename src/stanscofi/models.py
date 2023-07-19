#coding: utf-8

import numpy as np
import random
import pandas as pd
from scipy.sparse import coo_array, csr_array

from sklearn.decomposition import NMF as NonNegMatFact
from sklearn.linear_model import LogisticRegression as Logit

from .preprocessing import preprocessing_XY

###############################################################################################################
###################
# Basic model     #
###################

class BasicModel(object):
    '''
    A class used to encode a drug repurposing model

    ...

    Parameters
    ----------
    params : dict
        dictionary which contains method-wise parameters

    Attributes
    ----------
    name : str
        the name of the model
    model : depends on the implemented method
        may contain an instance of a class of sklearn classifiers
    ...
        other attributes might be present depending on the type of model

    Methods
    -------
    __init__(params)
        Initializes the model with preselected parameters
    fit(train_dataset, seed=1234)
        Preprocesses and fits the model 
    predict_proba(test_dataset)
        Outputs properly formatted predictions of the fitted model on test_dataset
    predict(scores)
        Applies the following decision rule: if score<threshold, then return the negative label, otherwise return the positive label
    recommend_k_pairs(dataset, k=1, threshold=None)
        Outputs the top-k (item, user) candidates (or candidates which score is higher than a threshold) in the input dataset
    print_scores(scores)
        Prints out information about scores
    print_classification(predictions)
        Prints out information about predicted labels
    preprocessing(train_dataset) [not implemented in BasicModel]
        Preprocess the input dataset into something that is an input to the self.model_fit if it exists
    model_fit(train_dataset) [not implemented in BasicModel]
        Fits the model on train_dataset
    model_predict_proba(test_dataset) [not implemented in BasicModel]
        Outputs predictions of the fitted model on test_dataset
    '''
    def __init__(self, params):
        '''
        Creates an instance of stanscofi.BasicModel

        ...

        Parameters
        ----------
        params : dict
            dictionary which contains method-wise parameters
        '''
        self.name = "Model"
        for param in params:
            setattr(self, param, params[param])

    def fit(self, train_dataset, seed=1234):
        '''
        Fitting the model on the training dataset.

        Not implemented in the BasicModel class.

        ...

        Parameters
        ----------
        train_dataset : stanscofi.Dataset
            training dataset on which the model should fit
        seed : int (default: 1234)
            random seed
        '''
        np.random.seed(seed)
        random.seed(seed)
        self.model_fit(*self.preprocessing(train_dataset, is_training=True))

    def predict_proba(self, test_dataset, default_zero_val=1e-31):
        '''
        Outputs properly formatted scores (not necessarily in [0,1]!) from the fitted model on test_dataset. Internally calls model_predict() then reformats the scores

        ...

        Parameters
        ----------
        test_dataset : stanscofi.Dataset
            dataset on which predictions should be made

        Returns
        ----------
        scores : COO-array of shape (n_items, n_users)
            sparse matrix in COOrdinate format, with nonzero values corresponding to predictions on available pairs in the dataset
        '''
        scores = self.model_predict_proba(*self.preprocessing(test_dataset, is_training=False))
        if ((scores!=0).any()):
            default_val = min(default_zero_val, np.min(scores[scores!=0])/2)
        else:
            default_val = default_zero_val
        #print(("folds",test_dataset.folds.data.shape[0]))
        if (scores.shape==test_dataset.folds.shape):
            scores[(scores==0)&(test_dataset.folds.toarray()==1)] = default_val ## avoid removing these zeroes
            scores = coo_array(scores)
            scores = scores*test_dataset.folds
            #print(("scores",scores.data.shape[0]))
            return coo_array(scores)
        assert scores.shape[0]==test_dataset.folds.data.shape[0]
        scores[(scores==0)&(test_dataset.folds.data==1)] = default_val ## avoid removing these zeroes 
        scores_arr = coo_array((scores, (test_dataset.folds.row, test_dataset.folds.col)), shape=test_dataset.folds.shape)
        #print(("scores",scores.data.shape[0]))
        return scores_arr

    def predict(self, scores, threshold=0.5):
        '''
        Outputs class labels based on the scores, using the following formula
            prediction = -1 if (score<threshold) else 1

        ...

        Parameters
        ----------
        scores : COO-array of shape (n_items, n_users)
            sparse matrix in COOrdinate format
        threshold : float
            the threshold of classification into the positive class

        Returns
        ----------
        predictions : COO-array of shape (n_items, n_users)
            sparse matrix in COOrdinate format with values in {-1,1}
        '''
        #print(("scores-preds",scores.data.shape[0]))
        preds = coo_array((scores.toarray()!=0).astype(int)*((-1)**(scores.toarray()<=threshold)))
        #print(('preds',preds.data.shape[0]))
        return preds

    def recommend_k_pairs(self, dataset, k=1, threshold=None):
        '''
        Outputs the top-k (item, user) candidates (or candidates which score is higher than a threshold) in the input dataset

        ...

        Parameters
        ----------
        dataset : stanscofi.Dataset
            dataset on which predictions should be made
        k : int or None (default: 1)
            number of pair candidates to return (with ties)
        threshold : float or None (default: 0)
            threshold on candidate scores. If k is not None, k best candidates are returned independently of the value of threshold
        ...

        Parameters
        ----------
        candidates : list of tuples of size 3
            list of (item, user, score) candidates (by name as present in the dataset)
        '''
        assert (k is None) or (k > 0)
        scores = self.predict_proba(dataset)
        if (k is not None):
            vals = scores
            kth_best = np.unique([scores.toarray()[a] for a in np.argsort(-scores)])[k]
            ids_list = np.argwhere(scores==kth_best).tolist()
        else:
            assert threshold is not None
            vals = (scores>=threshold)*scores
            k = len(vals.data)
            ids_list = np.dstack(np.unravel_index((-vals.toarray()).ravel().argsort(), scores.toarray().shape))[0][:k,:].tolist()
        candidates = [[dataset.item_list[i], dataset.user_list[j], scores.toarray()[i,j]] for i,j in ids_list]
        return candidates

    def print_scores(self, scores):
        '''
        Prints out information about the scores

        ...

        Parameters
        ----------
        scores : COO-array
            sparse matrix in COOrdinate format
        '''
        print("* Scores")
        print("%d unique items, %d unique users" % (len(np.unique(scores.row)), len(np.unique(scores.col))))
        print("Scores: Min: %f\tMean: %f\tMedian: %f\tMax: %f\tStd: %f\n" % tuple([f(scores.data) for f in [np.min,np.mean,np.median,np.max,np.std]]))

    def print_classification(self, predictions):
        '''
        Prints out information about the predicted classes

        ...

        Parameters
        ----------
        predictions : COO-array
            sparse matrix in COOrdinate format
        '''
        print("* Classification")
        print("%d unique items, %d unique users" % (len(np.unique(predictions.row)), len(np.unique(predictions.col))))
        print("Positive class: %d, Negative class: %d\n" % ((csr_array(predictions)==1).sum(), (csr_array(predictions)==-1).sum()))

    def preprocessing(self, dataset, is_training=True):
        '''
        Preprocessing step, which converts elements of a dataset (ratings matrix, user feature matrix, item feature matrix) into appropriate inputs to the classifier (e.g., X feature matrix for each (user, item) pair, y response vector).

        <Not implemented in the BasicModel class.>

        ...

        Parameters
        ----------
        dataset : stanscofi.Dataset
            dataset to convert
        is_training : bool
            is the preprocessing prior to training (true) or testing (false)?

        Returns
        ----------
        ... : ...
            appropriate inputs to the classifier (vary across algorithms)
        '''
        raise NotImplemented

    def model_fit(self):
        '''
        Fitting the model on the training dataset.

        <Not implemented in the BasicModel class.>

        ...

        Parameters
        ----------
        ... : ...
            appropriate inputs to the classifier (vary across algorithms)
        '''
        raise NotImplemented

    def model_predict_proba(self):
        '''
        Making predictions using the model on the testing dataset.

        <Not implemented in the BasicModel class.>

        ...

        Parameters
        ----------
        ... : ...
            appropriate inputs to the classifier (vary across algorithms)
        ...

        Returns
        ----------
        scores : array_like of shape (n_items, n_users)
            prediction values by the model
        '''
        raise NotImplemented

###############################################################################################################
###################
# NMF             #
###################

class NMF(BasicModel):
    '''
    Non-negative Matrix Factorization (calls sklearn.decomposition.NMF internally). It uses the very same parameters as sklearn.decomposition.NMF, so please refer to help(sklearn.decomposition.NMF).

    ...

    Parameters
    ----------
    params : dict
        dictionary which contains sklearn.decomposition.NMF parameters

    Attributes
    ----------
    Same as BasicModel class

    Methods
    -------
    Same as BasicModel class
    preprocessing(train_dataset)
        Preprocesses the input dataset into something that is an input to fit
    model_fit(train_dataset)
        Preprocesses and fits the model
    model_predict_proba(test_dataset)
        Outputs predictions of the fitted model on test_dataset
    '''
    def __init__(self, params):
        '''
        Creates an instance of stanscofi.NMF

        ...

        Parameters
        ----------
        params : dict
            dictionary which contains sklearn.decomposition.NMF parameters
        '''
        super(NMF, self).__init__(params)
        self.name = "NMF"
        self.estimator = NonNegMatFact(**params)

    def preprocessing(self, dataset, is_training=True):
        '''
        Preprocessing step, which converts elements of a dataset (ratings matrix, user feature matrix, item feature matrix) into appropriate inputs to the NMF classifier.

        ...

        Parameters
        ----------
        dataset : stanscofi.Dataset
            dataset to convert
        is_training : bool
            is the preprocessing prior to training (true) or testing (false)?

        Returns
        ----------
        args : contains
        A : array-like of shape (n_users, n_items)
            contains the transposed translated association matrix so that all its values are non-negative
        '''
        mat = np.nan_to_num(dataset.ratings.toarray(), copy=True, nan=0.)
        A = mat-np.min(mat)
        return [A.T]
    
    def model_fit(self, input):
        '''
        Fitting the NMF model on the preprocessed training dataset.

        ...

        Parameters
        ----------
        input : array-like of shape (n_samples,n_features)
            training data
        '''
        self.estimator.fit(input)
    
    def model_predict_proba(self, input):
        '''
        Making predictions using the NMF model on the testing dataset.

        ...

        Parameters
        ----------
        input : array-like of shape (n_samples,n_features)
            testing data
        ...

        Returns
        ----------
        result : array-like of shape (n_samples,n_features)
        '''
        W = self.estimator.fit_transform(input)
        return W.dot(self.estimator.components_).T

###############################################################################################################
#########################
# Logistic regression   #
#########################

class LogisticRegression(BasicModel):
    '''
    Logistic Regression (calls sklearn.linear_model.LogisticRegression internally). It uses the very same parameters as sklearn.linear_model.LogisticRegression, so please refer to help(sklearn.linear_model.LogisticRegression).

    ...

    Parameters
    ----------
    params : dict
        dictionary which contains sklearn.linear_model.LogisticRegression parameters, plus a key called "preprocessing" which determines which preprocessing function (in stanscofi.preprocessing) should be applied to data, plus a key called "subset" which gives the maximum number of features to consider in the model (those features will be the Top-subset in terms of variance across samples)

    Attributes
    ----------
    Same as BasicModel class

    Methods
    -------
    Same as BasicModel class
    preprocessing(train_dataset)
        Preprocesses the input dataset into something that is an input to fit
    model_fit(train_dataset)
        Preprocesses and fits the model
    model_predict_proba(test_dataset)
        Outputs predictions of the fitted model on test_dataset
    '''
    def __init__(self, params):
        '''
        Creates an instance of stanscofi.LogisticRegression

        ...

        Parameters
        ----------
        params : dict
            dictionary which contains sklearn.linear_model.LogisticRegression parameters, plus a key called "preprocessing_str" which determines which preprocessing function (in stanscofi.preprocessing) should be applied to data, plus a key called "subset" which gives the maximum number of features to consider in the model (those features will be the Top-subset in terms of variance across samples)
        '''
        super(LogisticRegression, self).__init__(params)
        assert self.preprocessing_str in ["Perlman_procedure", "meanimputation_standardize", "same_feature_preprocessing"]
        self.name = "LogisticRegression"
        self.scalerP, self.scalerS, self.filter = None, None, None
        self.estimator = Logit(**{p: params[p] for p in params if (p not in ["preprocessing_str", "subset"])})

    def preprocessing(self, dataset, is_training=True):
        '''
        Preprocessing step, which converts elements of a dataset (ratings matrix, user feature matrix, item feature matrix) into appropriate inputs to the Logistic Regression classifier. 

        ...

        Parameters
        ----------
        dataset : stanscofi.Dataset
            dataset to convert
        is_training : bool
            is the preprocessing prior to training (true) or testing (false)?

        Returns
        ----------
        args : contains
        X : array-like of shape (n_ratings, n_pair_features)
            (user, item) feature matrix (the actual contents of the matrix depends on parameters "preprocessing" and "subset" given as input
        y : array-like of shape (n_ratings, )
            response vector for each (user, item) pair
        '''
        X, y, scalerS, scalerP, filter_ = preprocessing_XY(dataset, self.preprocessing_str, subset_=self.subset, filter_=self.filter, scalerS=self.scalerS, scalerP=self.scalerP, inf=2, njobs=1)
        self.filter, self.scalerS, self.scalerP = filter_, scalerS, scalerP
        return [X, y] if (is_training) else [X]
    
    def model_fit(self, X, y):
        '''
        Fitting the Logistic Regression model on the training dataset.

        ...

        Parameters
        ----------
        X : array-like of shape (n_ratings, n_pair_features)
            (user, item) feature matrix (the actual contents of the matrix depends on parameters "preprocessing" and "subset" given as input
        y : array-like of shape (n_ratings, )
            response vector for each (user, item) pair
        '''
        self.estimator.fit(X, y)
    
    def model_predict_proba(self, X):
        '''
        Making predictions using the Logistic Regression model on the testing dataset.

        ...

        Parameters
        ----------
        X : array-like of shape (n_ratings, n_pair_features)
            (user, item) feature matrix (the actual contents of the matrix depends on parameters "preprocessing" and "subset" given as input
        '''
        scores = self.estimator.predict_proba(X)[:,1] ## scores for the positive class
        return scores