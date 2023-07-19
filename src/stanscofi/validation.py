#coding: utf-8

from sklearn.metrics import precision_recall_curve as PRC
from sklearn.metrics import roc_curve as ROC

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

###################################
## LIST OF AVAILABLE METRICS     ##
###################################
from cute_ranking.core import mean_reciprocal_rank, r_precision, precision_at_k, recall_at_k, f1_score_at_k, average_precision, mean_average_precision, dcg_at_k, ndcg_at_k, mean_rank, hit_rate_at_k
from sklearn.metrics import roc_auc_score, fbeta_score
from scipy.stats import kendalltau, spearmanr

def ERR(y_true, y_pred, max=10, max_grade=2):
    '''
        source: https://raw.githubusercontent.com/skondo/evaluation_measures/master/evaluations_measures.py
    '''
    max=10
    max_grade=2
    ranking = y_true[np.argsort(-y_pred)]
    if max is None:
        max = len(ranking)
    ranking = ranking[:min(len(ranking), max)]
    ranking = map(float, ranking)
    result = 0.0
    prob_step_down = 1.0 
    for rank, rel in enumerate(ranking):
        rank += 1
        utility = (pow(2, rel) - 1) / pow(2, max_grade)
        result += prob_step_down * utility / rank
        prob_step_down *= (1 - utility)  
    return result
AUC = lambda y_true, y_pred, k, u1 : roc_auc_score(y_true, y_pred, average="weighted")
Fscore = lambda y_true, y_pred, u, beta : fbeta_score(y_true, y_pred, beta=beta, average="weighted")
def TAU(y_true, y_pred, u, u1):
    if (len(np.unique(y_pred))==1):
        return 0
    res = kendalltau(y_true, y_pred)
    return res.correlation
def Rscore(y_true, y_pred, u, u1):
    if (len(np.unique(y_pred))==1):
        return 0
    res = spearmanr(y_true, y_pred)
    return res.correlation
MRR = lambda y_true, y_pred, u, u1 : mean_reciprocal_rank([y_true[np.argsort(-y_pred)]])
RP = lambda y_true, y_pred, u, u1 : r_precision(y_true[np.argsort(-y_pred)])
PrecisionK = lambda y_true, y_pred, k, u1 : precision_at_k(y_true[np.argsort(-y_pred)], k)
RecallK = lambda y_true, y_pred, k, u1 : recall_at_k(y_true[np.argsort(-y_pred)], np.sum(y_true>0), k=k)
def F1K(y_true, y_pred, k, u1):
    rec = recall_at_k(y_true[np.argsort(-y_pred)], np.sum(y_true>0), k)
    prec = precision_at_k(y_true[np.argsort(-y_pred)], k)
    if (rec+prec==0):
        return 0
    return f1_score_at_k(y_true[np.argsort(-y_pred)], np.sum(y_true>0), k=k)
AP = lambda y_true, y_pred, u, u1 : average_precision(y_true[np.argsort(-y_pred)])
MAP = lambda y_true, y_pred, u, u1 : mean_average_precision([y_true[np.argsort(-y_pred)]])
DCGk = lambda y_true, y_pred, k, u1 : dcg_at_k(y_true[np.argsort(-y_pred)], k)
NDCGk = lambda y_true, y_pred, k, u1 : ndcg_at_k(y_true[np.argsort(-y_pred)], k)
MeanRank = lambda y_true, y_pred, k, u1 : mean_rank([y_true[np.argsort(-y_pred)]])
HRk = lambda y_true, y_pred, k, u1 : hit_rate_at_k([y_true[np.argsort(-y_pred)]], k)
metrics_list = ["AUC", "Fscore", "TAU", "Rscore", "MRR", "RP", "PrecisionK", "RecallK", "F1K", "AP", "MAP", "DCGk", "NDCGk", "MeanRank", "HRk", "ERR"]

###################################
## COMPUTATION OF METRICS        ##
###################################

def compute_metrics(scores, predictions, dataset, metrics, k=1, beta=1, verbose=False):
    '''
    Computes *user-wise* validation metrics for a given set of scores and predictions w.r.t. a dataset

    ...

    Parameters
    ----------
    scores : COO-array of shape (n_items, n_users)
        sparse matrix in COOrdinate format
    predictions : COO-array of shape (n_items, n_users)
        sparse matrix in COOrdinate format with values in {-1,1}
    dataset : stanscofi.Dataset
        dataset on which the metrics should be computed
    metrics : lst of str
        list of metrics which should be computed
    k : int (default: 1)
        Argument of the metric to optimize. Implemented metrics are in validation.py
    beta : float (default: 1)
        Argument of the metric to optimize. Implemented metrics are in validation.py
    verbose : bool
        prints out information about ignored users for the computation of validation metrics, that is, users which pairs are only associated to a single class (i.e., all pairs with this users are either assigned 0, -1 or 1)

    Returns
    -------
    metrics : pandas.DataFrame of shape (len(metrics), 2)
        table of metrics: metrics in rows, average and standard deviation across users in columns
    plots_args : dict
        dictionary of arguments to feed to the plot_metrics function to plot the Precision-Recall and the Receiver Operating Chracteristic (ROC) curves
    '''
    metrics_list = ["AUC", "Fscore", "TAU", "Rscore", "MRR", "RP", "PrecisionK", "RecallK", "F1K", "AP", "MAP", "DCGk", "NDCGk", "MeanRank", "HRk", "ERR"]
    assert predictions.shape==scores.shape==dataset.folds.shape
    assert all([metric in metrics_list for metric in metrics])
    y_true_all = dataset.ratings.toarray()[dataset.folds.row,dataset.folds.col].ravel() 
    y_pred_all = predictions.data.ravel()
    scores_all = scores.data.ravel()
    assert y_true_all.shape==y_pred_all.shape==scores_all.shape
    ## Compute average metric per user
    user_ids = np.unique(dataset.folds.col)
    n_ignored = 0
    aucs, tprs, recs, fscores = [], [], [], []
    base_fpr = np.linspace(0, 1, 101)
    base_pres = np.linspace(0, 1, 101)
    metrics_list = {metric: [] for metric in metrics}
    for user_id in user_ids:
        user_ids_i = np.argwhere(dataset.folds.col==user_id)
        if (len(user_ids_i)==0):
            n_ignored += 1
            continue
        user_truth = y_true_all[user_ids_i]
        user_pred = y_pred_all[user_ids_i]
        if ((len(np.unique(user_truth))==2) and (1 in user_truth)):
            fpr, tpr, _ = ROC(user_truth, user_pred, pos_label=1.)
            pres, rec, _ = PRC(user_truth, user_pred)
            aucs.append(roc_auc_score(user_truth, user_pred, average="weighted"))
            fscores.append(fbeta_score(user_truth, user_pred, beta=beta, average="weighted"))
            tpr = np.interp(base_fpr, fpr, tpr)
            tpr[0] = 0.0
            tprs.append(tpr)
            rec = np.interp(base_pres, pres, rec)
            recs.append(rec)
            for metric in metrics:
                #print(metric)
                value = eval(metric)(user_truth.ravel(), user_pred.ravel(), k, beta)
                metrics_list.update({metric: metrics_list[metric]+[value]})
        else:
            n_ignored += 1
    if (verbose and n_ignored>0):
        print("<validation.compute_metrics> Computed on #users=%d, %d ignored (%2.f perc)" % (len(user_ids), n_ignored, 100*n_ignored/len(user_ids)))
    if (len(aucs)==0 or len(fscores)==0):
        metrics = pd.DataFrame([], index=metrics, 
		columns=["Average", "StandardDeviation"])
        return metrics, {}
    metrics = pd.DataFrame([[f(metrics_list[m]) for f in [np.mean, np.std]] for m in metrics_list], index=metrics, columns=["Average", "StandardDeviation"])
    metrics = pd.concat((metrics, pd.DataFrame([[k,beta]], index=["arguments (k, beta)"], columns=metrics.columns)), axis=0)
    return metrics, {"y_true": (y_true_all>0).astype(int), "y_pred": (y_pred_all>0).astype(int), "scores": scores_all, "predictions": y_pred_all, "ground_truth": y_true_all, "aucs": aucs, "fscores": fscores, "tprs": np.array(tprs), "recs": np.array(recs)}

###################################
## PLOTS                         ##
###################################

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
    base_fpr = np.linspace(0, 1, tprs.shape[1])
    base_pres = np.linspace(0, 1, np.array(recs).shape[1])
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
    axes[0,0].plot(base_fpr, mean_tprs, 'b', alpha = 0.8, label=model_name+' (AUC = %0.2f %s %0.2f)' % (auc, "$\\pm$", std_auc))
    axes[0,0].fill_between(base_fpr, tprs_lower, tprs_upper, color = 'blue', alpha = 0.2)
    axes[0,0].plot([0, 1], [0, 1], linestyle = '--', lw = 2, color = 'r', alpha= 0.8, label="Constant")
    axes[0,0].set_ylabel('True Positive Rate')
    axes[0,0].set_xlabel('False Positive Rate')
    axes[0,0].legend(loc="lower right")
    axes[0,0].set_title('Avg. user ROC curve')
    ## Precision-recall curve
    axes[0,1].plot(base_pres, mean_recs, 'b', alpha=0.8, label=model_name+' (F = %0.2f %s %0.2f)' % (fs, "$\\pm$", std_fs))
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