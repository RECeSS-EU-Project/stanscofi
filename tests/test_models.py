import unittest
import numpy as np
import pandas as pd

import stanscofi.datasets
import stanscofi.models

class TestModels(unittest.TestCase):

    ## Generate example
    def generate_dataset(self):
        npositive, nnegative, nfeatures, mean, std = 20, 10, 10, 0.5, 1
        data_args = stanscofi.datasets.generate_dummy_dataset(npositive, nnegative, nfeatures, mean, std)
        dataset = stanscofi.datasets.Dataset(**data_args)
        return dataset

    def test_scores2ratings(self):
        dataset = self.generate_dataset()
        preds = np.random.normal(0,1,size=np.prod(dataset.ratings_mat.shape)).reshape(dataset.ratings_mat.shape)
        ## matrix form
        scores = pd.DataFrame(preds, index=dataset.item_list, columns=dataset.user_list)
        ratings = stanscofi.models.scores2ratings(scores)
        self.assertEqual(ratings.shape[1], 3)
        self.assertEqual(ratings.shape[0], np.prod(scores.shape))
        self.assertEqual(ratings.values[0,2], scores.values[0,0])
        self.assertEqual(ratings.values[-1,2], scores.values[-1,-1])

    def test_create_scores(self):
        dataset = self.generate_dataset()
        ## unary form
        scores = stanscofi.models.create_scores(1, dataset)
        self.assertEqual(scores.shape[1], 3)
        self.assertEqual(scores.shape[0], np.prod(dataset.ratings_mat.shape))
        self.assertTrue((scores[:,2]==1).all())
        scores = stanscofi.models.create_scores(1., dataset)
        self.assertEqual(scores.shape[1], 3)
        self.assertEqual(scores.shape[0], np.prod(dataset.ratings_mat.shape))
        self.assertTrue((scores[:,2]==1.).all())
        ## vector form
        preds = np.random.normal(0,1,size=np.prod(dataset.ratings_mat.shape))
        scores = stanscofi.models.create_scores(preds, dataset)
        self.assertEqual(scores.shape[1], 3)
        self.assertEqual(scores.shape[0], preds.shape[0])
        self.assertTrue((scores[:,2]==preds).all())

    def test_create_overscores(self):
        dataset = self.generate_dataset()
        df = dataset.ratings.copy()
        df = df[:100,:]
        preds = np.random.normal(0,1,size=df.shape[0])
        preds[preds==0] = 1 
        df[:,2] = preds
        scores = stanscofi.models.create_overscores(df, dataset)
        self.assertEqual(scores.shape[1], 3)
        self.assertEqual(scores.shape[0], np.prod(dataset.ratings_mat.shape))
        self.assertEqual(np.sum(scores[:,2]!=0),preds.shape[0])

    def test_NMF(self):
        dataset = self.generate_dataset()
        params = {"init":None, "solver":'cd', "beta_loss":'frobenius', "tol":0.0001, "max_iter":100, 
          "random_state":12345, "alpha_W":0.0, "alpha_H":'same', "l1_ratio":0.0, "verbose":0, 
          "shuffle":False, "n_components": np.min(dataset.ratings_mat.shape)//2+1, "decision_threshold": 0.005}
        model = stanscofi.models.NMF(params)
        model.fit(dataset)
        scores = model.predict(dataset)
        predictions = model.classify(scores)
        ## if it ends without any error, it is a success

    def test_LogisticRegression(self):
        dataset = self.generate_dataset()
        params = {"penalty":'elasticnet', "C":1.0, "fit_intercept":True, "class_weight":"balanced", 
          "intercept_scaling":1., "random_state":12345, "max_iter":100, "tol": 1e-4, 
          "multi_class":'multinomial', "n_jobs": 1, "l1_ratio":1, "solver": "saga", 
          ## parameter subset allows to only consider Top-N features in terms of cross-sample variance for speed-up 
          "preprocessing": "meanimputation_standardize", "subset": 5, "decision_threshold": 0.75}
        model = stanscofi.models.LogisticRegression(params)
        model.fit(dataset)
        scores = model.predict(dataset)
        predictions = model.classify(scores)
        ## if it ends without any error, it is a success

if __name__ == '__main__':
    unittest.main()