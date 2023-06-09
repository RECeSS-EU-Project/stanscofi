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
        the list of scores where the first column correspond to users, second to items, third to ratings
    '''
    grid = np.argwhere(np.ones(df.shape))
    res_df = pd.DataFrame([], index=range(grid.shape[0]))
    res_df[user_col] = [df.columns[x] for x in list(grid[:, 1].flatten())]
    res_df[item_col] = [df.index[x] for x in list(grid[:, 0].flatten())]
    res_df[rating_col] = [df.values[i,j] for i,j in grid.tolist()]
    return res_df[[user_col, item_col, rating_col]]

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
        scores : stanscofi.Dataset
            dataset on which predictions should be made
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
        predictions = scores.copy()
        predictions[:,2] = (-1)**(predictions[:,2]<self.decision_threshold)
        return predictions.astype(int)

    def print_scores(self, scores):
        Nitems, Nusers = len(np.unique(scores[:, 1].tolist())), len(np.unique(scores[:, 0].tolist()))
        print("* Scores")
        print("%d unique items, %d unique users" % (Nitems, Nusers))
        print("Scores: Min: %f\tMean: %f\tMedian: %f\tMax: %f\tStd: %f\n" % tuple([f(scores[:, 2]) for f in [np.min,np.mean,np.median,np.max,np.std]]))

    def print_classification(self, predictions):
        Nitems, Nusers = len(np.unique(predictions[:, 1].tolist())), len(np.unique(predictions[:, 0].tolist()))
        print("* Classification")
        print("%d unique items, %d unique users" % (Nitems, Nusers))
        print("Positive class: %d, Negative class: %d\n" % (np.sum(predictions[:,2]==1), np.sum(predictions[:,2]==-1)))

    def preprocessing(self, train_dataset):
        raise NotImplemented

    def fit(self, train_dataset):
        raise NotImplemented

    def model_predict(self, test_dataset):
        raise NotImplemented

###############################################################################################################
###################
# NMF             #
###################

class NMF(BasicModel):
    def __init__(self, params):
        super(NMF, self).__init__(params)
        self.name = "NMF"
        self.model = NonNegMatFact(**{p: params[p] for p in params if (p!="decision_threshold")})

    def preprocessing(self, train_dataset):
        A_train = train_dataset.ratings_mat.copy()
        A_train -= np.min(A_train)
        return A_train.T
    
    def fit(self, train_dataset):
        inp = self.preprocessing(train_dataset)
        self.model.fit(inp)
    
    def model_predict(self, test_dataset):
        inp = self.preprocessing(test_dataset)
        W = self.model.fit_transform(inp)
        return W.dot(self.model.components_).T

###############################################################################################################
#########################
# Logistic regression   #
#########################

class LogisticRegression(BasicModel):
    def __init__(self, params):
        super(LogisticRegression, self).__init__(params)
        self.name = "LogisticRegression"
        self.scalerP, self.scalerS = None, None
        self.model = Logit(**{p: params[p] for p in params if (p not in ["decision_threshold", "preprocessing", "subset"])})
        self.preprocessing_str = params["preprocessing"]
        assert self.preprocessing_str in ["Perlman_procedure", "meanimputation_standardize", "same_feature_preprocessing"]
        self.subset = params["subset"]
        self.filter = None

    def preprocessing(self, train_dataset):
        if (self.preprocessing_str == "Perlman_procedure"):
            X, y = eval("stanscofi.preprocessing."+self.preprocessing_str)(train_dataset, njobs=1, sep_feature="-", missing=-666, verbose=False)
            scalerS, scalerP = None, None
        if (self.preprocessing_str == "meanimputation_standardize"):
            X, y, scalerS, scalerP = eval("stanscofi.preprocessing."+self.preprocessing_str)(train_dataset, subset=self.subset, scalerS=self.scalerS, scalerP=self.scalerP, inf=2, verbose=False)
        if (self.preprocessing_str == "same_feature_preprocessing"):
            X, y = eval("stanscofi.preprocessing."+self.preprocessing_str)(train_dataset)
            scalerS, scalerP = None, None
        if (self.preprocessing_str != "meanimputation_standardize"):
            if ((self.subset is not None) or (self.filter is not None)):
                if ((self.subset is not None) and (self.filter is None)):
                    with np.errstate(over="ignore"):
                        x_vars = [np.nanvar(X[:,i]) if (np.sum(~np.isnan(X[:,i]))>0) else 0 for i in range(X.shape[1])]
                        x_vars = [x if (not np.isnan(x) and not np.isinf(x)) else 0 for x in x_vars]
                        x_ids_vars = np.argsort(x_vars).tolist()
                        features = x_ids_vars[-self.subset:]
                        self.filter = features
                X = X[:,self.filter]
        self.scalerS = scalerS
        self.scalerP = scalerP
        return X, y
    
    def fit(self, train_dataset):
        X, y = self.preprocessing(train_dataset)
        self.model.fit(X, y)
    
    def model_predict(self, test_dataset):
        X, _ = self.preprocessing(test_dataset)
        preds = self.model.predict_proba(X)
        ids = np.argwhere(np.ones(test_dataset.ratings_mat.shape))
        predicted_ratings = np.zeros((X.shape[0], 3))
        predicted_ratings[:,0] = ids[:X.shape[0],1] 
        predicted_ratings[:,1] = ids[:X.shape[0],0] 
        predicted_ratings[:,2] = preds.max(axis=1)
        return predicted_ratings