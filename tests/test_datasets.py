import unittest
import numpy as np

import stanscofi.datasets
import stanscofi.utils

class TestDatasets(unittest.TestCase):

    def test_dummy_dataset(self):
        npositive, nnegative, nfeatures, mean, std = 200, 100, 50, 0.5, 1
        data_args = stanscofi.datasets.generate_dummy_dataset(npositive, nnegative, nfeatures, mean, std)
        self.assertTrue("ratings_mat" in data_args)
        self.assertTrue("users" in data_args)
        self.assertTrue("items" in data_args)
        self.assertEqual(len(data_args), 3)
        self.assertEqual(data_args["users"].shape[0], nfeatures//2)
        self.assertEqual(data_args["users"].shape[1], npositive+nnegative)
        self.assertEqual(data_args["items"].shape[0], nfeatures//2)
        self.assertEqual(data_args["items"].shape[1], npositive+nnegative)
        self.assertEqual(data_args["ratings_mat"].shape[0], npositive+nnegative)
        self.assertEqual(data_args["ratings_mat"].shape[1], npositive+nnegative)
        self.assertTrue(all([x in [-1,0,1] for x in np.unique(data_args["ratings_mat"].values)]))

    def test_existing_dataset(self):
        available_datasets = ["Gottlieb", "Cdataset", "DNdataset", "LRSSL", "PREDICT_Gottlieb", "TRANSCRIPT", "PREDICT"]
        values = {
                'Gottlieb': [593, 593, 313, 313, 1933, 0, 1.04],
                'Cdataset': [663, 663, 409, 409, 2532, 0, 0.93],
                'DNdataset': [550, 1490, 360, 4516, 1008, 0, 0.01],
                'LRSSL': [763, 1526, 681, 681, 3051, 0, 0.59],
                'PREDICT_Gottlieb': [593, 1779, 313, 313, 1933, 0, 1.04],
                'TRANSCRIPT': [204, 12096, 116, 12096, 401, 11, 0.45],
                'PREDICT': [1351, 6265, 1066, 2914, 5624, 152, 0.34],
        }
        for dataset_name in available_datasets:
            data_args = stanscofi.utils.load_dataset(dataset_name, save_folder="./")
            data_args.update({"name": dataset_name})
            if (dataset_name=="TRANSCRIPT"):
                data_args.update({"same_item_user_features": True})
            dataset = stanscofi.datasets.Dataset(**data_args)
            vals = values[dataset_name]
            self.assertEqual(dataset.items.shape[1], vals[0])
            self.assertEqual(dataset.items.shape[0], vals[1])
            self.assertEqual(dataset.users.shape[1], vals[2])
            self.assertEqual(dataset.users.shape[0], vals[3])
            self.assertEqual(dataset.ratings_mat.shape[1], vals[2])
            self.assertEqual(dataset.ratings_mat.shape[0], vals[0])
            self.assertEqual(np.sum(dataset.ratings_mat==1), vals[4])
            self.assertEqual(np.sum(dataset.ratings_mat==-1), vals[5])
            sparsity = np.sum(dataset.ratings_mat!=0)*100/np.prod(dataset.ratings_mat.shape)
            self.assertEqual(np.round(sparsity,2), vals[6])

    def test_new_dataset(self):
        npositive, nnegative, nfeatures, mean, std = 200, 100, 50, 0.5, 1
        data_args = stanscofi.datasets.generate_dummy_dataset(npositive, nnegative, nfeatures, mean, std)
        dataset = stanscofi.datasets.Dataset(ratings_mat=data_args["ratings_mat"], users=data_args["users"], items=data_args["items"])
        self.assertEqual(dataset.items.shape[0], nfeatures//2)
        self.assertEqual(dataset.items.shape[1], npositive+nnegative)
        self.assertEqual(dataset.users.shape[0], nfeatures//2)
        self.assertEqual(dataset.users.shape[1], npositive+nnegative)
        self.assertEqual(dataset.ratings_mat.shape[1], npositive+nnegative)
        self.assertEqual(dataset.ratings_mat.shape[0], npositive+nnegative)
        self.assertEqual(np.sum(dataset.ratings_mat==1), npositive**2)
        self.assertEqual(np.sum(dataset.ratings_mat==-1), nnegative**2)
        sparsity = np.sum(dataset.ratings_mat!=0)/np.prod(dataset.ratings_mat.shape)
        self.assertEqual(sparsity, (npositive**2+nnegative**2)/(npositive+nnegative)**2)

    def test_visualize(self):
        npositive, nnegative, nfeatures, mean, std = 200, 100, 50, 0.5, 1
        data_args = stanscofi.datasets.generate_dummy_dataset(npositive, nnegative, nfeatures, mean, std)
        dataset = stanscofi.datasets.Dataset(**data_args)
        dataset.visualize(withzeros=False)
        dataset.visualize(withzeros=True)
        ## Generate random class predictions
        pi=1/16
        npoints = dataset.ratings_mat.shape[0]*dataset.ratings_mat.shape[1]
        predictions = np.zeros((npoints, 3))
        predictions[:,0] = [i for i in range(dataset.ratings_mat.shape[1]) for _ in range(dataset.ratings_mat.shape[0])]
        predictions[:,1] = [j for _ in range(dataset.ratings_mat.shape[1]) for j in range(dataset.ratings_mat.shape[0])]
        predictions[:,2] = np.random.choice([-1,1], p=[pi,1-pi], size=npoints)
        dataset.visualize(predictions=predictions, withzeros=False)
        dataset.visualize(predictions=predictions, show_errors=True)
        ## if it ends without any error, it is a success

    def test_get_folds(self):
        npositive, nnegative, nfeatures, mean, std = 200, 100, 50, 0.5, 1
        data_args = stanscofi.datasets.generate_dummy_dataset(npositive, nnegative, nfeatures, mean, std)
        dataset = stanscofi.datasets.Dataset(**data_args)
        nitems, nusers = [x//3+1 for x in dataset.ratings_mat.shape]
        folds = np.array([[i,j,dataset.ratings_mat[i,j]] for i in range(nitems) for j in range(nusers)])
        subset = dataset.get_folds(folds)
        self.assertEqual(dataset.items.shape[0], nfeatures//2)
        self.assertEqual(dataset.items.shape[1], nusers+1)
        self.assertEqual(dataset.users.shape[0], nfeatures//2)
        self.assertEqual(dataset.users.shape[1], nitems+1)
        self.assertEqual(dataset.ratings_mat.shape[1], nusers+1)
        self.assertEqual(dataset.ratings_mat.shape[0], nitems+1)
        self.assertEqual(np.sum(dataset.ratings_mat==1), np.sum(folds[:,2]==1))
        self.assertEqual(np.sum(dataset.ratings_mat==-1), np.sum(folds[:,2]==-1))
        sparsity = np.sum(dataset.ratings_mat!=0)/np.prod(dataset.ratings_mat.shape)
        self.assertEqual(sparsity, np.sum(folds[:,2]!=0)/((nitems+1)*(nusers+1)))

if __name__ == '__main__':
    unittest.main()