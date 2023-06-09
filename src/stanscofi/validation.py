#coding: utf-8

from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import fbeta_score
from sklearn.metrics import precision_recall_curve as PRC
from sklearn.metrics import roc_curve as ROC
from scipy import interp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

def compute_metrics(scores, predictions, test_dataset, beta=1, ignore_zeroes=False, verbose=False):
    '''
    Computes validation metrics for a given set of scores and predictions w.r.t. a dataset

    ...

    Parameters
    ----------
    scores : array-like of shape (n_ratings, 3)
        a matrix which contains the user indices (column 1), the item indices (column 2) and the score for the corresponding (user, item) pair (float value in column 3)
    predictions : array-like of shape (n_ratings, 3)
        a matrix which contains the user indices (column 1), the item indices (column 2) and the class for the corresponding (user, item) pair (value in {-1, 0, 1} in column 3)
    test_dataset : stanscofi.Dataset
        the validation dataset on which the metrics should be computed
    beta : float
        a positive number for the computation of the F-beta metric
    ignore_zeroes : bool
        how to deal with more than two class labels; if ignore_zeroes=True, then pairs which are assigned 0 are not taken into account when computing the metric {{-1},{1}}; otherwise, zeroes are considered negative, and the validation metrics are computed {{-1,0},{1}}
    verbose : bool
        prints out information about ignored users for the computation of validation metrics, that is, users which pairs are only associated to a single class (i.e., all pairs with this users are either assigned 0, -1 or 1)

    Returns
    -------
    metrics : pandas.DataFrame of shape (2, 2)
        table of metrics: AUC, F_beta score in rows, average and standard deviation across users in columns
    plots_args : dict
        dictionary of arguments to feed to the plot_metrics function
    '''
    assert scores.shape[1]==3
    assert predictions.shape[1]==3
    assert predictions.shape[0]==scores.shape[0]
    assert beta>0
    y_true_all = np.array([test_dataset.true_values.get((i,j), 0) for i,j in scores[:,:2].astype(int).tolist()])
    y_pred_all = predictions[:,2].flatten()
    if (not ignore_zeroes):
        scores_ = deepcopy(scores)
        predictions_ = deepcopy(predictions)
        y_true = (y_true_all>0).astype(int) 
        y_pred = (y_pred_all>0).astype(int)
    else:
        ids = np.argwhere(y_true!=0)
        scores_ = scores[ids.flatten().tolist(),:]
        predictions_ = predictions[ids,:]
        y_true = (y_true_all[ids]>0).astype(int)
        y_pred = (y_pred_all[ids]>0).astype(int)
    assert y_true.shape[0]==y_pred.shape[0]
    assert all([x in [-1,0,1] for x in np.unique(y_true).flatten()])
    ## Compute average metric per user
    user_ids = np.unique(scores_[:,0].flatten()).astype(int).tolist()
    n_ignored = 0
    aucs, tprs, recs, fscores = [], [], [], []
    base_fpr = np.linspace(0, 1, 101)
    base_pres = np.linspace(0, 1, 101)
    for user_id in user_ids:
        user_ids_i = np.argwhere(scores_[:,0].flatten()==user_id)
        user_truth = y_true[user_ids_i].reshape(1, -1)
        user_pred = y_pred[user_ids_i].reshape(1, -1)
        if (len(np.unique(user_truth))==2):
            fpr, tpr, _ = ROC(user_truth.flatten(), user_pred.flatten())
            pres, rec, _ = PRC(user_truth.flatten(), user_pred.flatten())
            aucs.append(AUC(user_truth.flatten(), user_pred.flatten()))
            fscores.append(fbeta_score(user_truth.flatten(), user_pred.flatten(), beta=beta))
            tpr = interp(base_fpr, fpr, tpr)
            tpr[0] = 0.0
            tprs.append(tpr)
            rec = interp(base_pres, pres, rec)
            recs.append(rec)
        else:
            n_ignored += 1
    if (verbose and n_ignored>0):
        print("<validation.compute_metrics> Computed on #users=%d, %d ignored (%2.f perc)" % (len(user_ids), n_ignored, 100*n_ignored/len(user_ids)))
    metrics = pd.DataFrame([[f(x) for f in [np.mean, np.std]] for x in [aucs, fscores]], index=["AUC", "F_%.1f" % beta], 
		columns=["Avg. across users", "Std"])
    return metrics, {"y_true": y_true, "y_pred": y_pred, "scores": scores_[:,2].flatten(), "predictions": y_pred_all, "ground_truth": y_true_all, "aucs": aucs, "fscores": fscores, "tprs": np.array(tprs), "recs": np.array(recs)}

def plot_metrics(y_true=None, y_pred=None, scores=None, ground_truth=None, predictions=None, aucs=None, fscores=None, tprs=None, recs=None, figsize=(16,5), model_name="Model"):
    '''
    Plots the ROC curve, the Precision-Recall curve, the boxplot of predicted scores and the piechart of classes associated to the predictions y_pred in input w.r.t. ground truth y_true

    ...

    Parameters
    ----------
    y_true : array-like of shape (n_ratings,)
        an array which contains the binary ground truth labels in {0,1}
    y_pred : array-like of shape (n_ratings,)
        an array which contains the binary predicted labels in {0,1}
    scores : array-like of shape (n_ratings,)
        an array which contains the predicted scores
    ground_truth : array-like of shape (n_ratings,)
        an array which contains the ground truth labels in {-1,0,1}
    predictions : array-like of shape (n_ratings,)
        an array which contains the predicted labels in {-1,0,1}
    aucs : list
        list of AUCs per user
    fscores : list
        list of F-scores per user
    tprs : array-like of shape (n_thresholds,)
        Increasing true positive rates such that element i is the true positive rate of predictions with score >= thresholds[i], where thresholds was given as input to sklearn.metrics.roc_curve
    recs : array-like of shape (n_thresholds,)
        Decreasing recall values such that element i is the recall of predictions with score >= thresholds[i] and the last element is 0, where thresholds was given as input to sklearn.metrics.precision_recall_curve
    figsize : tuple of size 2
        width and height of the figure
    model_name : str
        model which predicted the ratings

    Returns
    -------
    metrics : pandas.DataFrame of shape (2, 2)
        table of metrics: AUC, F_beta score in rows, average and standard deviation across users in columns
    plots_args : dict
        dictionary of arguments to feed to the plot_metrics function
    '''
    assert y_true.shape[0]==y_pred.shape[0]==scores.shape[0]
    assert ((y_true==1)|(y_true==0)).all()
    assert ((y_pred==1)|(y_pred==0)).all()
    assert ground_truth.shape[0]==predictions.shape[0]==scores.shape[0]
    assert ((ground_truth==1)|(ground_truth==0)|(ground_truth==-1)).all()
    assert ((predictions==1)|(predictions==0)|(predictions==-1)).all()
    assert len(aucs)==len(fscores)
    assert tprs.shape[0]==recs.shape[0]
    assert len(figsize)==2
    base_fpr = np.linspace(0, 1, 101)
    base_pres = np.linspace(0, 1, 101)
    ## Compute average values across users
    if (len(aucs) > 0):
        mean_tprs = tprs.mean(axis=0)
        std_tprs = tprs.std(axis=0)
        recs = np.array(recs)
        mean_recs = recs.mean(axis=0)
        std_recs = recs.std(axis=0)
        auc = np.mean(aucs)
        std_auc = np.std(aucs)
        fs = np.mean(fscores)
        std_fs = np.std(fscores)
    else:
        auc, std_auc = [np.nan]*2
        print("<validation.plot_metrics> Can't plot: only 1 relevance level in true scores")
        return None
    tprs_upper = np.minimum(mean_tprs + std_tprs, 1)
    tprs_lower = mean_tprs - std_tprs
    recs_upper = np.minimum(mean_recs + std_recs, 1)
    recs_lower = mean_recs - std_recs
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
    ## ROC curve
    axes[0,0].plot(base_fpr, mean_tprs, 'b', alpha = 0.8, label=model_name+' (AUC = %0.2f $\pm$ %0.2f)' % (auc, std_auc))
    axes[0,0].fill_between(base_fpr, tprs_lower, tprs_upper, color = 'blue', alpha = 0.2)
    axes[0,0].plot([0, 1], [0, 1], linestyle = '--', lw = 2, color = 'r', alpha= 0.8, label="Constant")
    axes[0,0].set_ylabel('True Positive Rate')
    axes[0,0].set_xlabel('False Positive Rate')
    axes[0,0].legend(loc="lower right")
    axes[0,0].set_title('Avg. user ROC curve')
    ## Precision-recall curve
    axes[0,1].plot(base_pres, mean_recs, 'b', alpha=0.8, label=model_name+' (F = %0.2f $\pm$ %0.2f)' % (fs, std_fs))
    axes[0,1].fill_between(base_pres, recs_lower, recs_upper, color="blue", alpha=0.2)
    axes[0,1].plot([0,1], [1,0], linestyle="--", lw=2, color="r", alpha=0.8, label="Constant")
    axes[0,1].set_xlabel('Precision')
    axes[0,1].set_ylabel('Recall')
    axes[0,1].set_title('Avg. user precision-recall curve')
    axes[0,1].legend(loc='lower left')
    ## Boxplot of predicted values
    boxes = [
        {
            'label' : "Score",
            'whislo': np.percentile(scores, 10),    # Bottom whisker position
            'q1'    : np.percentile(scores, 25),    # First quartile (25th percentile)
            'med'   : np.percentile(scores, 50),    # Median         (50th percentile)
            'q3'    : np.percentile(scores, 75),    # Third quartile (75th percentile)
            'whishi': np.percentile(scores, 90),    # Top whisker position
            'fliers': []        # Outliers
        }
    ]
    axes[1,0].bxp(boxes, showfliers=False)
    labels, counts = np.unique(np.multiply(predictions, ground_truth)+2*ground_truth, return_counts=True)
    ids = np.argwhere(labels!=0).flatten().tolist()
    ## Piechart of predicted labels
    h = axes[1,1].pie(counts[ids], labels=[{3:'TP', -3:'FP', 0:'UN', 1: "FN", -1: "TN"}[x] for x in labels[ids]], autopct='%1.1f%%', colors=[{3:'green', -3:'red', 0:'white', 1: "darkred", -1: "darkgreen"}[x] for x in labels[ids]])
    accuracy = 0 if (3 not in labels) else counts[labels.tolist().index(3)] 
    accuracy += 0 if (-1 not in labels) else counts[labels.tolist().index(-1)] 
    accuracy /= np.sum(counts[ids])
    axes[1,1].text(-1., -1.25, "%d datapoints (%d known matchings)\nAccuracy=%.2f on known matchings" % (np.sum(counts), np.sum(counts[ids]), accuracy))
    ## Show
    plt.show()