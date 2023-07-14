import unittest
import numpy as np
from scipy.sparse import coo_array

import sys
sys.path.insert(0,"../src/")
from stanscofi.datasets import generate_dummy_dataset, Dataset
from stanscofi.validation import metrics_list, compute_metrics, plot_metrics

class TestValidation(unittest.TestCase):

    ## Generate example
    def generate_dataset_scores_threshold(self):
        threshold=0.5
        npositive, nnegative, nfeatures, mean, std = 200, 100, 50, 0.5, 1
        data_args = generate_dummy_dataset(npositive, nnegative, nfeatures, mean, std)
        dataset = Dataset(**data_args)
        ## Generate random class scores
        np.random.seed(1223435)
        pi=1/16
        npoints = np.sum(dataset.folds.data)
        scores = np.random.normal(np.random.choice([-10,10], p=[pi,1-pi], size=npoints), 1).reshape(dataset.folds.shape)
        return dataset, coo_array(scores), threshold

    def test_compute_metrics(self):
        dataset, scores, threshold = self.generate_dataset_scores_threshold()
        predictions = coo_array((-1)**(scores.toarray()<threshold))
        metrics, _ = compute_metrics(scores, predictions, dataset, metrics=tuple(metrics_list), k=1, beta=1, verbose=False)
        print(metrics)
        self.assertEqual(metrics.shape[0], len(metrics_list)+1)
        self.assertEqual(metrics.shape[1], 2)
        self.assertEqual(np.round(metrics.loc["AUC"]["Average"],1), 0.5)
        self.assertEqual(np.round(metrics.loc["AUC"]["StandardDeviation"],1), 0.0)
        self.assertEqual(np.round(metrics.loc["Fscore"]["Average"],1), 0.5)
        self.assertEqual(np.round(metrics.loc["Fscore"]["StandardDeviation"],1), 0.0)

    def test_plot_metrics(self):
        dataset, scores, threshold = self.generate_dataset_scores_threshold()
        predictions = coo_array((-1)**(scores.toarray()<threshold))
        _, plot_args = compute_metrics(scores, predictions, dataset, metrics=("AUC", "MRR"), k=1, beta=1, verbose=False)
        plot_metrics(**plot_args, figsize=(10,10), model_name="Random on Dummy")
        ## if it ends without any error, it is a success

if __name__ == '__main__':
    unittest.main()