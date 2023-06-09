import unittest
import numpy as np

import stanscofi.datasets
import stanscofi.models
import stanscofi.training_testing

class TestTrainingTesting(unittest.TestCase):

    ## Generate example
    def generate_dataset_folds(self):
        npositive, nnegative, nfeatures, mean, std = 20, 10, 6, 0.5, 1
        data_args = stanscofi.datasets.generate_dummy_dataset(npositive, nnegative, nfeatures, mean, std)
        dataset = stanscofi.datasets.Dataset(**data_args)
        nitems, nusers = [x//3+1 for x in dataset.ratings_mat.shape]
        folds = np.array([[i,j,dataset.ratings_mat[i,j]] for i in range(nitems) for j in range(nusers)])
        subset = dataset.get_folds(folds)
        return dataset, folds, subset

    def test_print_folds(self):
        dataset, folds, subset = self.generate_dataset_folds()
        stanscofi.training_testing.print_folds(folds, dataset, fold_name="Fold")
        ## if it ends without any error, it is a success

    def test_traintest_validation_split(self):
        ##### disjoint_users=False
        dataset, folds, subset = self.generate_dataset_folds()
        test_size = 0.1
        train_set, test_set, v1, v2  = stanscofi.training_testing.traintest_validation_split(dataset, test_size, early_stop=None, metric="cityblock", disjoint_users=False, random_state=1234, verbose=False, print_dists=False)
        ## are items disjoints and weakly correlated?
        self.assertEqual(sum([np.unique(X[:,1]).shape[0] for X in [train_set,test_set]]), np.unique(dataset.ratings[:,1]).shape[0])
        ## are all ratings classified?
        self.assertEqual(sum([X.shape[0] for X in [train_set,test_set]]), dataset.ratings.shape[0])
        ## test size is respected
        self.assertTrue(int(test_size*dataset.ratings.shape[0])>=test_set.shape[0])
        self.assertEqual(len(v1), 0)
        self.assertEqual(len(v2), 0)
        ##### disjoint_users=True
        train_set, test_set, val1_set, val2_set = stanscofi.training_testing.traintest_validation_split(dataset, test_size, early_stop=None, metric="cityblock", disjoint_users=True, random_state=1234, verbose=False, print_dists=False)
        ## are user disjoints?
        self.assertEqual(sum([np.unique(X[:,0]).shape[0] for X in [train_set, test_set]]), np.unique(dataset.ratings[:,0]).shape[0])
        ## are items disjoints and weakly correlated?
        self.assertTrue(sum([np.unique(X[:,1]).shape[0] for X in [train_set,test_set]])<=np.unique(dataset.ratings[:,1]).shape[0])
        ## are all ratings classified?
        self.assertEqual(sum([X.shape[0] for X in [train_set, test_set, val1_set, val2_set] if (len(X)>0)]), dataset.ratings.shape[0])
        ## test size is respected
        self.assertTrue(int(test_size*dataset.ratings.shape[0])>=test_set.shape[0])

    def test_cv_training(self):
        dataset, _, _ = self.generate_dataset_folds()
        params = {"init":None, "solver":'cd', "beta_loss":'frobenius', "tol":0.0001, "max_iter":100, 
          "random_state":12345, "alpha_W":0.0, "alpha_H":'same', "l1_ratio":0.0, "verbose":0, 
          "shuffle":False, "n_components": np.min(dataset.ratings_mat.shape)//2+1, "decision_threshold": 0.005}
        template = stanscofi.models.NMF
        ## no parallel
        best_estimator_no_parallel = stanscofi.training_testing.cv_training(template, params, dataset, metric="AUC", beta=1, njobs=1, nsplits=5, random_state=1234, show_plots=True, verbose=False)
        auc_test, auc_train, model_params, cv_folds = [best_estimator_no_parallel[x] for x in ["test_AUC", "train_AUC", "model_params", "cv_folds"]]
        ## parallel
        best_estimator_parallel = stanscofi.training_testing.cv_training(template, params, dataset, metric="AUC", beta=1, njobs=4, nsplits=5, random_state=1234, show_plots=True, verbose=False)
        auc_test_, auc_train_, model_params_, cv_folds_ = [best_estimator_parallel[x] for x in ["test_AUC", "train_AUC", "model_params", "cv_folds"]]
        self.assertEqual(np.round(auc_test,1), np.round(auc_test_,1))
        self.assertEqual(np.round(auc_train,1), np.round(auc_train_,1))

    def test_grid_search(self):
        dataset, _, _ = self.generate_dataset_folds()
        search_params = {"n_components": range(2, np.min(dataset.ratings_mat.shape)//2, 5)}
        params = {"init":None, "solver":'cd', "beta_loss":'frobenius', "tol":0.0001, "max_iter":100, 
          "random_state":12345, "alpha_W":0.0, "alpha_H":'same', "l1_ratio":0.0, "verbose":0, 
          "shuffle":False, "n_components": np.min(dataset.ratings_mat.shape)//2+1, "decision_threshold": 0.005}
        template = stanscofi.models.NMF
        ## no parallel
        best_params_no_parallel, best_estimator_no_parallel = stanscofi.training_testing.grid_search(search_params, template, params, dataset, metric="AUC", njobs=1, nsplits=5, random_state=1234, show_plots=True, verbose=False)
        auc_test, auc_train, model_params, cv_folds = [best_estimator_no_parallel[x] for x in ["test_AUC", "train_AUC", "model_params", "cv_folds"]]
        ## parallel
        best_params_parallel, best_estimator_parallel = stanscofi.training_testing.grid_search(search_params, template, params, dataset, metric="AUC", njobs=4, nsplits=5, random_state=1234, show_plots=True, verbose=False)
        auc_test_, auc_train_, model_params_, cv_folds_ = [best_estimator_parallel[x] for x in ["test_AUC", "train_AUC", "model_params", "cv_folds"]]
        self.assertEqual(np.round(auc_test,1), np.round(auc_test_,1))
        self.assertEqual(np.round(auc_train,1), np.round(auc_train_,1))
        for p in best_params_no_parallel:
            self.assertEqual(best_params_no_parallel[p], best_params_parallel[p])

if __name__ == '__main__':
    unittest.main()