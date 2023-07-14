import unittest
import numpy as np
import pandas as pd

import sys
sys.path.insert(0,"../src/")
from stanscofi.datasets import generate_dummy_dataset, Dataset
from stanscofi.models import NMF, LogisticRegression

class TestModels(unittest.TestCase):

    ## Generate example
    def generate_dataset(self):
        npositive, nnegative, nfeatures, mean, std = 20, 10, 10, 0.5, 1
        data_args = generate_dummy_dataset(npositive, nnegative, nfeatures, mean, std)
        dataset = Dataset(**data_args)
        return dataset

    def test_NMF(self):
        dataset = self.generate_dataset()
        params = {"init":None, "solver":'cd', "beta_loss":'frobenius', "tol":0.0001, "max_iter":100, 
          "random_state":12345, "alpha_W":0.0, "alpha_H":'same', "l1_ratio":0.0, "verbose":0, 
          "shuffle":False, "n_components": np.min(dataset.ratings.shape)//2+1}
        model = NMF(params)
        model.fit(dataset, seed=params["random_state"])
        scores = model.predict_proba(dataset)
        predictions = model.predict(scores, threshold=0)
        ## if it ends without any error, it is a success

    def test_LogisticRegression(self):
        dataset = self.generate_dataset()
        params = {"penalty":'elasticnet', "C":1.0, "fit_intercept":True, "class_weight":"balanced", 
          "intercept_scaling":1., "random_state":12345, "max_iter":100, "tol": 1e-4, 
          "multi_class":'multinomial', "n_jobs": 1, "l1_ratio":1, "solver": "saga", 
          ## parameter subset allows to only consider Top-N features in terms of cross-sample variance for speed-up 
          "preprocessing_str": "meanimputation_standardize", "subset": 5}
        model = LogisticRegression(params)
        model.fit(dataset, seed=params["random_state"])
        scores = model.predict_proba(dataset)
        predictions = model.predict(scores, threshold=0)
        ## if it ends without any error, it is a success

    def test_print(self):
        dataset = self.generate_dataset()
        params = {"init":None, "solver":'cd', "beta_loss":'frobenius', "tol":0.0001, "max_iter":100, 
          "random_state":12345, "alpha_W":0.0, "alpha_H":'same', "l1_ratio":0.0, "verbose":0, 
          "shuffle":False, "n_components": np.min(dataset.ratings.shape)//2+1}
        model = NMF(params)
        model.fit(dataset, seed=params["random_state"])
        dataset.summary()
        scores = model.predict_proba(dataset)
        model.print_scores(scores)
        predictions = model.predict(scores, threshold=0)
        model.print_classification(predictions) 
        lst_k = model.recommend_k_pairs(dataset, k=1) 
        lst_thres = model.recommend_k_pairs(dataset, k=None, threshold=lst_k[0][-1])
        self.assertEqual(len(lst_k), len(lst_thres))

if __name__ == '__main__':
    unittest.main()