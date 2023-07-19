import unittest
import numpy as np
from scipy.sparse import coo_array

import sys
sys.path.insert(0,"../src/")
from stanscofi.datasets import generate_dummy_dataset, Dataset
from stanscofi.utils import load_dataset

class TestDatasets(unittest.TestCase):

    def generate_dataset(self):
        npositive, nnegative, nfeatures, mean, std = 200, 100, 50, 0.5, 1
        data_args = generate_dummy_dataset(npositive, nnegative, nfeatures, mean, std)
        dataset = Dataset(**data_args)
        return dataset, data_args

    def test_dummy_dataset(self):
        npositive, nnegative, nfeatures, mean, std = 200, 100, 50, 0.5, 1
        data_args = generate_dummy_dataset(npositive, nnegative, nfeatures, mean, std)
        self.assertTrue("ratings" in data_args)
        self.assertTrue("users" in data_args)
        self.assertTrue("items" in data_args)
        self.assertEqual(len(data_args), 3)
        self.assertEqual(data_args["users"].shape[0], nfeatures//2)
        self.assertEqual(data_args["users"].shape[1], npositive+nnegative)
        self.assertEqual(data_args["items"].shape[0], nfeatures//2)
        self.assertEqual(data_args["items"].shape[1], npositive+nnegative)
        self.assertEqual(data_args["ratings"].shape[0], npositive+nnegative)
        self.assertEqual(data_args["ratings"].shape[1], npositive+nnegative)
        self.assertTrue(all([x in [-1,0,1] for x in np.unique(data_args["ratings"].toarray())]))

    def test_existing_dataset(self):
        ## For PREDICT, considering the publicly (partial) dataset on Zenodo
        available_datasets = ["Gottlieb", "Cdataset", "DNdataset", "LRSSL", "PREDICT_Gottlieb", "TRANSCRIPT", "PREDICT", "TRANSCRIPT_v1", "PREDICT_v1"]
        values = {
                'Gottlieb': [593, 593, 313, 313, 1933, 0, 1.04],
                'Cdataset': [663, 663, 409, 409, 2532, 0, 0.93],
                'DNdataset': [550, 1490, 360, 4516, 1008, 0, 0.01],
                'LRSSL': [763, 2049, 681, 681, 3051, 0, 0.59],
                'PREDICT_Gottlieb': [593, 1779, 313, 313, 1933, 0, 1.04],
                'TRANSCRIPT': [204, 12096, 116, 12096, 401, 11, 0.45],
                #'PREDICT': [1351, 6265, 1066, 2914, 5624, 152, 0.34], ##private version
                'PREDICT': [1014, 1642, 941, 1490, 4627, 132, 0.40], ##public version
                'TRANSCRIPT_v1': [558, 10811, 118, 10811, 773, 181, 0.76],
                #'PREDICT_v1': [1395, 6030, 1501, 2361, 8240, 295, 0.38], ##private version
                'PREDICT_v1': [20, 2150, 58, 1170, 59, 3, 0.28], ##public version
        }
        for dataset_name in available_datasets:
            data_args = load_dataset(dataset_name, save_folder="./")
            #data_args = load_dataset(dataset_name, save_folder="../datasets")
            data_args.update({"name": dataset_name})
            if (dataset_name=="TRANSCRIPT"):
                data_args.update({"same_item_user_features": True})
            dataset = Dataset(**data_args)
            ndrugs, ndiseases, ndrugs_known, ndiseases_known, npositive, nnegative, nunlabeled_unavailable, _, sparsity, sparsity_known, ndrug_features, missing_drug_features, ndisease_features, missing_disease_features = dataset.summary()
            vals = values[dataset_name]
            self.assertEqual(ndrugs_known, vals[0])
            self.assertEqual(ndiseases_known, vals[2])
            self.assertEqual(ndrug_features, vals[1])
            self.assertEqual(ndisease_features, vals[3])
            self.assertEqual(npositive, vals[4])
            self.assertEqual(nnegative, vals[5])
            self.assertEqual(np.round(sparsity,2), vals[6])

    def test_new_dataset(self):
        npositive, nnegative, nfeatures, mean, std = 200, 100, 50, 0.5, 1
        data_args = generate_dummy_dataset(npositive, nnegative, nfeatures, mean, std)
        dataset = Dataset(ratings=data_args["ratings"], users=data_args["users"], items=data_args["items"])
        self.assertEqual(dataset.items.shape[0], nfeatures//2)
        self.assertEqual(dataset.items.shape[1], npositive+nnegative)
        self.assertEqual(dataset.users.shape[0], nfeatures//2)
        self.assertEqual(dataset.users.shape[1], npositive+nnegative)
        self.assertEqual(dataset.ratings.shape[1], npositive+nnegative)
        self.assertEqual(dataset.ratings.shape[0], npositive+nnegative)
        self.assertEqual(np.sum(dataset.ratings==1), npositive**2)
        self.assertEqual(np.sum(dataset.ratings==-1), nnegative**2)
        sparsity = np.sum(dataset.ratings!=0)/np.prod(dataset.ratings.shape)
        self.assertEqual(sparsity, (npositive**2+nnegative**2)/(npositive+nnegative)**2)

    def test_visualize(self):
        dataset, _ = self.generate_dataset()
        dataset.visualize(withzeros=False)
        dataset.visualize(withzeros=False, dimred_args={"n_neighbors":10}) ## UMAP
        dataset.visualize(withzeros=True)
        ## Generate random class predictions
        pi=1/16
        npoints = np.prod(dataset.ratings.shape)
        predictions = coo_array(np.random.choice([-1,1], p=[pi,1-pi], size=np.prod(dataset.folds.shape)))
        dataset.visualize(predictions=predictions, withzeros=False)
        dataset.visualize(predictions=predictions, show_errors=True)
        ## if it ends without any error, it is a success

    def test_subset(self):
        npositive, nnegative, nfeatures, mean, std = 200, 100, 50, 0.5, 1
        data_args = generate_dummy_dataset(npositive, nnegative, nfeatures, mean, std)
        dataset = Dataset(**data_args)
        nitems, nusers = [x//3+1 for x in dataset.ratings.toarray().shape]
        ids = np.array([[i,j] for i in range(nitems) for j in range(nusers)])
        data = [1]*(nusers*nitems)
        folds = coo_array((data, (ids[:,0].ravel(), ids[:,1].ravel())), shape=dataset.folds.shape)
        subset = dataset.subset(folds)
        self.assertEqual(subset.nitem_features, nfeatures//2)
        self.assertEqual(len(subset.folds.data), nusers*nitems)
        self.assertEqual(subset.nuser_features, nfeatures//2)
        self.assertEqual(np.sum(subset.ratings==1), np.sum((dataset.ratings*subset.folds)==1))
        self.assertEqual(np.sum(subset.ratings==-1), np.sum((dataset.ratings*subset.folds)==-1))
        with self.assertRaises(ValueError):
            subset = dataset.subset(np.array([])) # no dataset should be created, and a warning should be sent

if __name__ == '__main__':
    unittest.main()