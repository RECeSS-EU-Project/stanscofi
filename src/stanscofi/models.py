#coding: utf-8

import numpy as np
import pandas as pd

from sklearn.decomposition import NMF as NonNegMatFact
from sklearn.linear_model import LogisticRegression as Logit

import stanscofi.preprocessing

###############################################################################################################
###################
# Utils           #
###################

def scores2ratings(df, user_col="user", item_col="item", rating_col="rating"):
    '''
    Converts a matrix of scores into a list of scores

    ...

    Parameters
    ----------
    df : pandas.DataFrame of shape (n_items, n_users)
        the matrix of scores
    user_col : str
        column denoting users
    item_col : str
        column denoting items
    rating_col : str
        column denoting ratings in {-1, 0, 1}

    Returns
    ----------
    ratings : pandas.DataFrame of shape (n_ratings, 3)
        the list of scores where the first column correspond to users, second to items, third to scores
    '''
    grid = np.argwhere(np.ones(df.shape))
    res_df = pd.DataFrame([], index=range(grid.shape[0]))
    res_df[user_col] = [df.columns[x] for x in list(grid[:, 1].flatten())]
    res_df[item_col] = [df.index[x] for x in list(grid[:, 0].flatten())]
    res_df[rating_col] = [df.values[i,j] for i,j in grid.tolist()]
    return res_df[[user_col, item_col, rating_col]]

def create_scores(preds, dataset):
    '''
    Converts a score vector or a score value into a list of scores

    ...

    Parameters
    ----------
    preds : int or float or array-like of shape (n_ratings, )
        the matrix of scores
    dataset : stanscofi.datasets.Dataset
        dataset to apply the scores to

    Returns
    ----------
    scores : array-like of shape (n_ratings, 3)
        the list of scores where the first column correspond to users, second to items, third to scores
    '''
    assert str(type(preds)) in ["<class 'float'>", "<class 'int'>"] or len(preds.shape)==1 or preds.shape[1]==1
    ids = np.argwhere(np.ones(dataset.ratings_mat.shape))
    assert str(type(preds)) in ["<class 'float'>", "<class 'int'>"] or preds.shape[0]==ids.shape[0]
    scores = np.zeros((ids.shape[0], 3))
    scores[:,0] = ids[:,1] 
    scores[:,1] = ids[:,0] 
    scores[:,2] = np.ravel(preds)
    return scores

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
        dictionary which contains method-wise parameters plus a key called "decision_threshold" with a float value which determines the decision threshold to label a positive class

    Attributes
    ----------
    name : str
        the name of the model
    model : depends on the implemented method
        may contain an instance of a class of sklearn classifiers
    decision_threshold : float
        decision threshold to label a positive class
    ...
        other attributes might be present depending on the type of model

    Methods
    -------
    __init__(params)
        Initialize the model with preselected parameters
    to_picklable()
        Outputs a dictionary which contains all attributes of the model
    from_picklable(picklable)
        Sets all parameters in the model according to dictionary picklable
    predict(test_dataset)
        Outputs properly formatted predictions of the fitted model on test_dataset
    classify(scores)
        Applies the following decision rule: if score<self.decision_threshold, then return -1, otherwise return 1
    print_scores(scores)
        Prints out information about scores
    print_classification(predictions)
        Prints out information about predicted labels
    preprocessing(train_dataset)
        Preprocess the input dataset into something that is an input to the self.model.fit if it exists (not implemented in BasicModel)
    fit(train_dataset)
        Preprocess and fit the model (not implemented in BasicModel)
    model_predict(test_dataset)
        Outputs predictions of the fitted model on test_dataset (not implemented in BasicModel)
    '''
    def __init__(self, params):
        '''
        Creates an instance of stanscofi.BasicModel

        ...

        Parameters
        ----------
        params : dict
            dictionary which contains method-wise parameters plus a key called "decision_threshold" with a float value which determines the decision threshold to label a positive class
        '''
        self.name = "Model"
        self.model = None
        assert "decision_threshold" in params
        self.decision_threshold = params["decision_threshold"]

    def to_picklable(self):
        '''
        Gets a serializable version of the model

        ...

        Returns
        ----------
        picklable : dict
            dictionary which contains all attributes of the model
        '''
        objs = [x for x in dir(self.model) 
		if (x[0]!="_" and x[-1]!="_" and str(type(x))!="<class 'method'>")
        ]
        picklable = {}
        for a in objs:
            picklable.setdefault(a, eval("self.model."+a))
        picklable.setdefault("decision_threshold", self.decision_threshold)
        return picklable

    def from_picklable(self, picklable):
        '''
        Reinitializes a model based on its serializable version

        ...

        Parameters
        ----------
        picklable : dict
            dictionary which contains all attributes of the model
        '''
        for a in picklable:
            if (str(type(picklable[a]))!="<class 'method'>"):
                setattr(self.model, a, picklable[a])

    def predict(self, test_dataset):
        '''
        Outputs properly formatted predictions of the fitted model on test_dataset. Internally calls model_predict() then reformats the predictions

        ...

        Parameters
        ----------
        test_dataset : stanscofi.Dataset
            dataset on which predictions should be made

        Returns
        ----------
        scores : array-like of shape (n_ratings, 3)
            the list of scores where the first column correspond to users, second to items, third to scores
        '''
        preds = self.model_predict(test_dataset)
        if (preds.shape==test_dataset.ratings_mat.shape):
            scores_mat = pd.DataFrame(preds, index=test_dataset.item_list, columns=test_dataset.user_list)
            ratings = scores2ratings(scores_mat, user_col="user", item_col="item", rating_col="rating")
            ratings["user"] = [test_dataset.user_list.index(x) for x in ratings["user"]]
            ratings["item"] = [test_dataset.item_list.index(x) for x in ratings["item"]]
            scores = ratings[["user","item","rating"]].values
        else:
            scores = preds
        return scores

    def classify(self, scores):
        '''
        Outputs class labels based on the scores, using the following formula
            prediction = -1 if (score<self.decision_threshold) else 1

        ...

        Parameters
        ----------
        scores : array-like of shape (n_ratings, 3)
            the list of scores where the first column correspond to users, second to items, third to scores

        Returns
        ----------
        predictions : array-like of shape (n_ratings, 3)
            the list of scores where the first column correspond to users, second to items, third to ratings
        '''
        predictions = scores.copy()
        predictions[:,2] = (-1)**(predictions[:,2]<self.decision_threshold)
        return predictions.astype(int)

    def print_scores(self, scores):
        '''
        Prints out information about the scores

        ...

        Parameters
        ----------
        scores : array-like of shape (n_ratings, 3)
            the list of scores where the first column correspond to users, second to items, third to scores
        '''
        Nitems, Nusers = len(np.unique(scores[:, 1].tolist())), len(np.unique(scores[:, 0].tolist()))
        print("* Scores")
        print("%d unique items, %d unique users" % (Nitems, Nusers))
        print("Scores: Min: %f\tMean: %f\tMedian: %f\tMax: %f\tStd: %f\n" % tuple([f(scores[:, 2]) for f in [np.min,np.mean,np.median,np.max,np.std]]))

    def print_classification(self, predictions):
        '''
        Prints out information about the predicted classes

        ...

        Parameters
        ----------
        predictions : array-like of shape (n_ratings, 3)
            the list of scores where the first column correspond to users, second to items, third to ratings
        '''
        Nitems, Nusers = len(np.unique(predictions[:, 1].tolist())), len(np.unique(predictions[:, 0].tolist()))
        print("* Classification")
        print("%d unique items, %d unique users" % (Nitems, Nusers))
        print("Positive class: %d, Negative class: %d\n" % (np.sum(predictions[:,2]==1), np.sum(predictions[:,2]==-1)))

    def preprocessing(self, dataset):
        '''
        Preprocessing step, which converts elements of a dataset (ratings matrix, user feature matrix, item feature matrix) into appropriate inputs to the classifier (e.g., X feature matrix for each (user, item) pair, y response vector).

        Not implemented in the BasicModel class.

        ...

        Parameters
        ----------
        dataset : stanscofi.Dataset
            dataset to convert

        Returns
        ----------
        ... : ...
            appropriate inputs to the classifier (vary across algorithms)
        '''
        raise NotImplemented

    def fit(self, train_dataset):
        '''
        Fitting the model on the training dataset.

        Not implemented in the BasicModel class.

        ...

        Parameters
        ----------
        train_dataset : stanscofi.Dataset
            training dataset on which the model should fit
        '''
        raise NotImplemented

    def model_predict(self, test_dataset):
        '''
        Making predictions using the model on the testing dataset.

        Not implemented in the BasicModel class.

        ...

        Parameters
        ----------
        test_dataset : stanscofi.Dataset
            testing dataset on which the model should be validated
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
        dictionary which contains sklearn.decomposition.NMF parameters plus a key called "decision_threshold" with a float value which determines the decision threshold to label a positive class

    Attributes
    ----------
    Same as BasicModel class

    Methods
    -------
    Same as BasicModel class
    preprocessing(train_dataset)
        Preprocess the input dataset into something that is an input to the self.model.fit if it exists
    fit(train_dataset)
        Preprocess and fit the model
    model_predict(test_dataset)
        Outputs predictions of the fitted model on test_dataset
    '''
    def __init__(self, params):
        '''
        Creates an instance of stanscofi.NMF

        ...

        Parameters
        ----------
        params : dict
            dictionary which contains sklearn.decomposition.NMF parameters plus a key called "decision_threshold" with a float value which determines the decision threshold to label a positive class
        '''
        super(NMF, self).__init__(params)
        self.name = "NMF"
        self.model = NonNegMatFact(**{p: params[p] for p in params if (p!="decision_threshold")})

    def preprocessing(self, dataset):
        '''
        Preprocessing step, which converts elements of a dataset (ratings matrix, user feature matrix, item feature matrix) into appropriate inputs to the NMF classifier.

        ...

        Parameters
        ----------
        dataset : stanscofi.Dataset
            dataset to convert

        Returns
        ----------
        A : array-like of shape (n_users, n_items)
            transposed translated association matrix so that all its values are non-negative
        '''
        A = dataset.ratings_mat.copy()
        A -= np.min(A)
        return A.T
    
    def fit(self, train_dataset):
        '''
        Fitting the NMF model on the training dataset.

        ...

        Parameters
        ----------
        train_dataset : stanscofi.Dataset
            training dataset on which the model should fit
        '''
        inp = self.preprocessing(train_dataset)
        self.model.fit(inp)
    
    def model_predict(self, test_dataset):
        '''
        Making predictions using the NMF model on the testing dataset.

        ...

        Parameters
        ----------
        test_dataset : stanscofi.Dataset
            testing dataset on which the model should be validated
        '''
        inp = self.preprocessing(test_dataset)
        W = self.model.fit_transform(inp)
        return W.dot(self.model.components_).T

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
        dictionary which contains sklearn.linear_model.LogisticRegression parameters plus a key called "decision_threshold" with a float value which determines the decision threshold to label a positive class, plus a key called "preprocessing" which determines which preprocessing function (in stanscofi.preprocessing) should be applied to data, plus a key called "subset" which gives the maximum number of features to consider in the model (those features will be the Top-subset in terms of variance across samples)

    Attributes
    ----------
    Same as BasicModel class

    Methods
    -------
    Same as BasicModel class
    preprocessing(train_dataset)
        Preprocess the input dataset into something that is an input to the self.model.fit if it exists
    fit(train_dataset)
        Preprocess and fit the model
    model_predict(test_dataset)
        Outputs predictions of the fitted model on test_dataset
    '''
    def __init__(self, params):
        '''
        Creates an instance of stanscofi.LogisticRegression

        ...

        Parameters
        ----------
        params : dict
            dictionary which contains sklearn.linear_model.LogisticRegression parameters plus a key called "decision_threshold" with a float value which determines the decision threshold to label a positive class, plus a key called "preprocessing" which determines which preprocessing function (in stanscofi.preprocessing) should be applied to data, plus a key called "subset" which gives the maximum number of features to consider in the model (those features will be the Top-subset in terms of variance across samples)
        '''
        super(LogisticRegression, self).__init__(params)
        self.name = "LogisticRegression"
        self.scalerP, self.scalerS = None, None
        self.model = Logit(**{p: params[p] for p in params if (p not in ["decision_threshold", "preprocessing", "subset"])})
        self.preprocessing_str = params["preprocessing"]
        assert self.preprocessing_str in ["Perlman_procedure", "meanimputation_standardize", "same_feature_preprocessing"]
        self.subset = params["subset"]
        self.filter = None

    def preprocessing(self, dataset):
        '''
        Preprocessing step, which converts elements of a dataset (ratings matrix, user feature matrix, item feature matrix) into appropriate inputs to the Logistic Regression classifier. 

        ...

        Parameters
        ----------
        dataset : stanscofi.Dataset
            dataset to convert

        Returns
        ----------
        X : array-like of shape (n_ratings, n_pair_features)
            (user, item) feature matrix (the actual contents of the matrix depends on parameters "preprocessing" and "subset" given as input
        y : array-like of shape (n_ratings, )
            response vector for each (user, item) pair
        '''
        X, y, scalerS, scalerP, filter_ = stanscofi.preprocessing.preprocessing_routine(dataset, self.preprocessing_str, subset_=self.subset, filter_=self.filter, scalerS=self.scalerS, scalerP=self.scalerP, inf=2, njobs=1)
        self.filter = filter_
        self.scalerS = scalerS
        self.scalerP = scalerP
        return X, y
    
    def fit(self, train_dataset):
        '''
        Fitting the Logistic Regression model on the training dataset.

        ...

        Parameters
        ----------
        train_dataset : stanscofi.Dataset
            training dataset on which the model should fit
        '''
        X, y = self.preprocessing(train_dataset)
        self.model.fit(X, y)
    
    def model_predict(self, test_dataset):
        '''
        Making predictions using the Logistic Regression model on the testing dataset.

        ...

        Parameters
        ----------
        test_dataset : stanscofi.Dataset
            testing dataset on which the model should be validated
        '''
        X, _ = self.preprocessing(test_dataset)
        preds = self.model.predict_proba(X).max(axis=1)
        scores = create_scores(preds, test_dataset)
        return scores