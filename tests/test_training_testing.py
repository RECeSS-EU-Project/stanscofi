import unittest
import numpy as np
from scipy.sparse import coo_array

import sys
sys.path.insert(0,"../src/")
from stanscofi.datasets import generate_dummy_dataset, Dataset
from stanscofi.training_testing import weakly_correlated_split, grid_search, cv_training
from stanscofi.models import NMF

class TestTrainingTesting(unittest.TestCase):

    ## Generate example
    def generate_dataset_folds(self):
        npositive, nnegative, nfeatures, mean, std = 200, 100, 10, 0.5, 1
        data_args = generate_dummy_dataset(npositive, nnegative, nfeatures, mean, std)
        dataset = Dataset(**data_args)
        nitems, nusers = [x//3+1 for x in dataset.ratings.toarray().shape]
        ids = np.array([[i,j] for i in range(nitems) for j in range(nusers)])
        folds = coo_array(([1]*(nusers*nitems), (ids[:,0].ravel(), ids[:,1].ravel())), shape=dataset.folds.shape)
        subset = dataset.subset(folds)
        return dataset, folds, subset

    def test_weakly_correlated_split(self):
        dataset, _, _ = self.generate_dataset_folds()
        test_size = 0.3
        (train_folds, test_folds), (dist_train_test, dist_train, dist_test) = weakly_correlated_split(dataset, test_size, early_stop=None, metric="euclidean")
        self.assertEqual(train_folds.shape, dataset.folds.shape)
        self.assertEqual(test_folds.shape, dataset.folds.shape)
        ## are items weakly correlated?
        self.assertTrue(dist_train_test>=max(dist_train, dist_test))
        ## are items disjoints? in the union of training and testing sets?
        self.assertEqual(train_folds.data.sum()+test_folds.data.sum(),dataset.folds.data.sum())
        ## test size is respected
        self.assertTrue(int(test_size*dataset.folds.data.sum())>=test_folds.data.sum())
        subset_train = dataset.subset(train_folds)
        self.assertEqual(subset_train.folds.data.sum(), train_folds.data.sum())
        subset_test = dataset.subset(test_folds)
        self.assertEqual(subset_test.folds.data.sum(), test_folds.data.sum())

    def test_cv_training(self):
        dataset, _, _ = self.generate_dataset_folds()
        params = {"init":None, "solver":'cd', "beta_loss":'frobenius', "tol":0.0001, "max_iter":100, 
          "random_state":12345, "alpha_W":0.0, "alpha_H":'same', "l1_ratio":0.0, "verbose":0, 
          "shuffle":False, "n_components": np.min(dataset.ratings.shape)//2+1}
        template = NMF
        ## no parallel
        results_no_parallel = cv_training(template, params, dataset, threshold=0, metric="AUC", beta=1, njobs=1, nsplits=3, random_state=1234, cv_type="random", show_plots=False, verbose=False)
        [m.predict_proba(dataset) for m in results_no_parallel["models"]]
        ## parallel
        results_parallel = cv_training(template, params, dataset, threshold=0, metric="AUC", beta=1, njobs=2, nsplits=3, random_state=1234, cv_type="random", show_plots=False, verbose=False)
        self.assertEqual(np.round(np.max(results_no_parallel["test_metric"]),1), np.round(np.max(results_parallel["test_metric"]),1))
        self.assertEqual(np.round(np.max(results_no_parallel["train_metric"]),1), np.round(np.max(results_parallel["train_metric"]),1))
        ## no parallel
        results_no_parallel = cv_training(template, params, dataset, threshold=0, metric="AUC", beta=1, njobs=1, nsplits=3, random_state=1234, cv_type="weakly_correlated", early_stop=1, show_plots=False, verbose=False)
        [m.predict_proba(dataset) for m in results_no_parallel["models"]]
        ## parallel
        results_parallel = cv_training(template, params, dataset, threshold=0, metric="AUC", beta=1, njobs=2, nsplits=3, random_state=1234, cv_type="weakly_correlated", early_stop=1, show_plots=False, verbose=False)
        self.assertEqual(np.round(np.max(results_no_parallel["test_metric"]),1), np.round(np.max(results_parallel["test_metric"]),1))
        self.assertEqual(np.round(np.max(results_no_parallel["train_metric"]),1), np.round(np.max(results_parallel["train_metric"]),1))

    def test_grid_search(self):
        dataset, _, _ = self.generate_dataset_folds()
        search_params = {
            "n_components": [2, np.min(dataset.ratings.shape)//3, np.min(dataset.ratings.shape)//2],
        }
        params = {"init":None, "solver":'cd', "beta_loss":'frobenius', "tol":0.0001, "max_iter":100, 
          "random_state":12345, "alpha_W":0.0, "alpha_H":'same', "l1_ratio":0.0, "verbose":0, 
          "shuffle":False, "n_components": np.min(dataset.ratings.shape)//3}
        template = NMF
        ## no parallel
        best_params_no_parallel, best_estimator_no_parallel, best_metrics_no_parallel = grid_search(search_params, template, params, dataset,  threshold=0, metric="AUC", njobs=1, nsplits=3, random_state=1234, show_plots=False, verbose=False)
        ## parallel
        best_params_parallel, best_estimator_parallel, best_metrics_parallel = grid_search(search_params, template, params, dataset, threshold=0,  metric="AUC", njobs=2, nsplits=3, random_state=1234, show_plots=False, verbose=False)
        self.assertEqual(np.round(best_metrics_no_parallel["test_metric"],1), np.round(best_metrics_parallel["test_metric"],1))
        self.assertEqual(np.round(best_metrics_no_parallel["train_metric"],1), np.round(best_metrics_parallel["train_metric"],1))
        for p in best_params_no_parallel:
            self.assertEqual(best_params_no_parallel[p], best_params_parallel[p])

if __name__ == '__main__':
    unittest.main()