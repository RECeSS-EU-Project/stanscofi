import unittest
import numpy as np

import stanscofi.validation

class TestValidation(unittest.TestCase):

    ## Generate example
    def generate_dataset_scores_threshold(self):
        threshold=0.5
        npositive, nnegative, nfeatures, mean, std = 200, 100, 50, 0.5, 1
        data_args = generate_dummy_dataset(npositive, nnegative, nfeatures, mean, std)
        dataset = stanscofi.datasets.Dataset(**data_args)
        ## Generate random class scores
        pi=1/16
        npoints = dataset.ratings_mat.shape[0]*dataset.ratings_mat.shape[1]
        scores = np.zeros((npoints, 3))
        scores[:,0] = [i for i in range(dataset.ratings_mat.shape[1]) for _ in range(dataset.ratings_mat.shape[0])]
        scores[:,1] = [j for _ in range(dataset.ratings_mat.shape[1]) for j in range(dataset.ratings_mat.shape[0])]
        scores[:,2] = np.random.normal(np.random.choice([-10,10], p=[pi,1-pi], size=npoints), 1).tolist()
        scores[:,2] -= np.min(scores[:,2])
        scores[:,2] /= np.max(scores[:,2])
        return dataset, scores, threshold

    def test_compute_metrics(self):
        dataset, scores, threshold = self.generate_dataset_scores_threshold()
        predictions = np.copy(scores)
        predictions[:,2] = (-1)**(predictions[:,2]<threshold)
        metrics, _ = compute_metrics(scores, predictions, dataset, beta=1, ignore_zeroes=False, verbose=False)
        print(metrics)
        self.assertEqual(metrics.shape[0], 2)
        self.assertEqual(metrics.shape[1], 2)

    def test_plot_metrics(self):
        dataset, scores, threshold = self.generate_dataset_scores_threshold()
        predictions = np.copy(scores)
        predictions[:,2] = (-1)**(predictions[:,2]<threshold)
        _, plot_args = compute_metrics(scores, predictions, dataset, beta=1, ignore_zeroes=False, verbose=False)
        stanscofi.validation.plot_metrics(**plot_args, figsize=(10,10), model_name="Random on Dummy")
        ## if it ends without any error, it is a success

if __name__ == '__main__':
    unittest.main()