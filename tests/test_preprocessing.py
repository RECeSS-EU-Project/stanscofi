#coding: utf-8

import unittest
import numpy as np
from tqdm import tqdm

import stanscofi.datasets
import stanscofi.preprocessing

## Non-optimized, naive implementation of Perlman's procedure
def Perlman_procedure_slow(dataset, sep_feature="-", missing=-666, inf=2, verbose=False):
    is_user = sum([int(sep_feature in str(x)) for x in dataset.user_features])
    is_item = sum([int(sep_feature in str(x)) for x in dataset.item_features])
    assert is_user in [0, dataset.users.shape[0]]
    assert is_item in [0, dataset.items.shape[0]]
    A, P, S = dataset.ratings_mat.copy(), dataset.users.copy(), dataset.items.copy()
    np.nan_to_num(P, copy=False, nan=0, posinf=inf, neginf=-inf)
    np.nan_to_num(S, copy=False, nan=0, posinf=inf, neginf=-inf)
    ## Ensure positive values in P and S
    P = P-np.tile(np.nanmin(P, axis=1), (P.shape[1], 1)).T
    S = S-np.tile(np.nanmin(S, axis=1), (S.shape[1], 1)).T
    y = np.ravel(A.flatten())
    ## All item,user pairs (items x users)
    ids = np.argwhere(np.ones(A.shape)) 
    ## All types of features (length=#features)
    nuser_features = [missing]*len(dataset.user_features) if (is_user==0) else np.array([x.split(sep_feature)[0] for x in dataset.user_features])
    nitem_features = [missing]*len(dataset.item_features) if (is_item==0) else np.array([x.split(sep_feature)[0] for x in dataset.item_features])
    ## All item/user corresponding to each type of feature (length=#features)
    ## In subset datasets some features might relate to absent diseases/drugs
    nuser_nms = [dataset.user_list.index(x) if (x in dataset.user_list) else missing for x in (dataset.user_features if (is_user==0) else [y.split(sep_feature)[1] for y in dataset.user_features])]
    nitem_nms = [dataset.item_list.index(x) if (x in dataset.item_list) else missing for x in (dataset.item_features if (is_item==0) else [y.split(sep_feature)[1] for y in dataset.item_features])]
    Nuf, Nif = len(set(nuser_features)), len(set(nitem_features))
    if (verbose):
        print("<preprocessing.Perlman_procedure> %d item similarities, %d user similarities" % (Nuf, Nif))
    X = np.zeros((ids.shape[0], Nuf*Nif)) ## (item, user) pairs x feature type
    known = {(i,u): int(A[i,u]!=0) for i in nitem_nms for u in nuser_nms if ((i!=missing) and (u!=missing))}
    if (verbose):
        print("<preprocessing.Perlman_procedure> For %d ratings pairs, identified %d features" % X.shape)
    for ff, [fi, fu] in enumerate([[a,b] for a in set(nitem_features) for b in set(nuser_features)]):
        if (verbose):
            print("<preprocessing.Perlman_procedure> Considering item feature '%s' and user feature '%s'" % (str(fi) if (fi!=missing) else "{1}", str(fu) if (fu!=missing) else "{1}"))
        for ii, [i, u] in tqdm(enumerate(ids.tolist())):
            ## f(i, dr') x f'(di', u) for all item feature f, user feature f', item dr', user di'
            all_sim_item = [( ifi, S[ifi,i] ) for ifi in range(S.shape[0])]
            ## fi(i, dr') x f'(di', u) for all user feature f', item dr', user di'
            all_sim_item = [( ifi, x*int(nitem_features[ifi]==fi) ) for ifi, x in all_sim_item]
            ## fi(i, dr') x f'(di', u) for all user feature f', item dr'!=i, user di'
            all_sim_item = [( ifi, x*int(nitem_nms[ifi]!=i) ) for ifi, x in all_sim_item]
            ## fi(i, dr') x fu(di', u) for all item dr'!=i, user di'
            all_sim_user = [( ufu, P[ufu,u] ) for ufu in range(P.shape[0])]
            all_sim_user = [( ufu, x*int(nuser_features[ufu]==fu) ) for ufu, x in all_sim_user]
            ## fi(i, dr') x fu(di', u) for all item dr'!=i, user di'!=u
            all_sim_user = [( ufu, x*int(nuser_nms[ufu]!=u) ) for ufu, x in all_sim_user]
            all_sim = [[ifi, ufu, all_sim_item[ifi][-1]*all_sim_user[ufu][-1]] for ifi in range(S.shape[0]) for ufu in range(P.shape[0])]
            ## fi(i, dr') x fu(di', u) for all item dr'!=i, user di'!=u and rating(dr',di')!=0
            X[ii,ff] = np.max([x*known.get((nitem_nms[ifi], nuser_nms[ufu]), 0) for ifi, ufu, x in all_sim])
    return np.sqrt(X), y

class TestPreprocessing(unittest.TestCase):

    ## Generate example
    def generate_dataset(self, same_item_user_features=False, sep_feature="--", ntype_feature=2):
        npositive, nnegative, mean, std = 4, 4, 0.5, 1
        nfeatures = (npositive+nnegative)*ntype_feature*2
        data_args = stanscofi.datasets.generate_dummy_dataset(npositive, nnegative, nfeatures, mean, std)
        if (len(sep_feature)!=0):
            assert ntype_feature%2==0
            data_args["users"].index = ["feature%d%s%s" % (f, sep_feature, u) for f in range(1,ntype_feature+1) for u in data_args["users"].columns]
            data_args["items"].index = ["feature%d%s%s" % (f, sep_feature, i) for f in range(1,ntype_feature+1) for i in data_args["items"].columns]
        data_args.setdefault("same_item_user_features", same_item_user_features)
        dataset = stanscofi.datasets.Dataset(**data_args)
        return dataset

    def test_meanimputation_standardize(self):
        dataset = self.generate_dataset(ntype_feature=2)
        X, y, scalerS, scalerP = stanscofi.preprocessing.meanimputation_standardize(dataset, subset=None, scalerS=None, scalerP=None, inf=int(1e1), verbose=False)
        self.assertEqual(y.shape[0], X.shape[0])
        self.assertEqual(y.shape[0], np.prod(dataset.ratings_mat.shape))
        ## imputation
        self.assertTrue((np.isfinite(X)).all())
        self.assertTrue((~np.isnan(X)).all())
        ## standard
        self.assertTrue((np.isclose(np.var(X, axis=0),1)).all())
        self.assertTrue((np.isclose(np.mean(X, axis=0),0)).all())
        self.assertTrue(all([x in [-1,0,1] for x in np.unique(y)]))
        ## subset=None
        self.assertEqual(X.shape[1], dataset.items.shape[0]+dataset.users.shape[0])
        X2, y2, _, _ = stanscofi.preprocessing.meanimputation_standardize(dataset, subset=None, scalerS=scalerS, scalerP=scalerP, inf=int(1e1), verbose=False)
        self.assertEqual(y.shape[0], X.shape[0])
        self.assertEqual(y.shape[0], np.prod(dataset.ratings_mat.shape))
        ## imputation
        self.assertTrue((np.isfinite(X)).all())
        self.assertTrue((~np.isnan(X)).all())
        ## standard
        self.assertTrue((np.isclose(np.var(X, axis=0),1)).all())
        self.assertTrue((np.isclose(np.mean(X, axis=0),0)).all())
        self.assertTrue(all([x in [-1,0,1] for x in np.unique(y)]))
        ## subset=None
        self.assertEqual(X.shape[1], dataset.items.shape[0]+dataset.users.shape[0])
        self.assertTrue((y2==y).all())
        self.assertTrue((X2==X).all())
        subset=2
        X, y, _, _ = stanscofi.preprocessing.meanimputation_standardize(dataset, subset=subset, scalerS=None, scalerP=None, inf=int(1e1), verbose=False)
        self.assertEqual(y.shape[0], X.shape[0])
        self.assertEqual(y.shape[0], np.prod(dataset.ratings_mat.shape))
        ## imputation
        self.assertTrue((np.isfinite(X)).all())
        self.assertTrue((~np.isnan(X)).all())
        ## standard
        self.assertTrue((np.isclose(np.var(X, axis=0),1)).all())
        self.assertTrue((np.isclose(np.mean(X, axis=0),0)).all())
        self.assertTrue(all([x in [-1,0,1] for x in np.unique(y)]))
        ## subset!=None
        self.assertEqual(X.shape[1], subset*2)
        dataset.visualize(X=X, y=y, withzeros=False)

    def test_same_feature_preprocessing(self):
        dataset = self.generate_dataset(same_item_user_features=True,ntype_feature=2)
        X, y = stanscofi.preprocessing.same_feature_preprocessing(dataset)
        self.assertEqual(y.shape[0], X.shape[0])
        self.assertEqual(y.shape[0], np.prod(dataset.ratings_mat.shape))
        ## same features
        self.assertEqual(X.shape[1], dataset.items.shape[0])
        self.assertEqual(X.shape[1], dataset.users.shape[0])
        dataset.visualize(X=X, y=y, withzeros=False)
        with self.assertRaises(AssertionError):
            dataset = self.generate_dataset(same_item_user_features=False)
            X, y = stanscofi.preprocessing.same_feature_preprocessing(dataset)

    def test_Perlman_procedure(self):
        ntype_feature = 2
        dataset = self.generate_dataset(sep_feature="--", ntype_feature=ntype_feature)
        X_true, y_true = Perlman_procedure_slow(dataset, sep_feature="--")
        self.assertEqual(y_true.shape[0], X_true.shape[0])
        self.assertEqual(y_true.shape[0], np.prod(dataset.ratings_mat.shape))
        self.assertEqual(X_true.shape[1], ntype_feature**2)
        X, y = stanscofi.preprocessing.Perlman_procedure(dataset, njobs=1, sep_feature="--")
        self.assertEqual(y.shape[0], X.shape[0])
        self.assertEqual(y.shape[0], np.prod(dataset.ratings_mat.shape))
        self.assertEqual(X.shape[1], ntype_feature**2)
        self.assertEqual(X.shape[1], X_true.shape[1])
        self.assertEqual(X.shape[0], X_true.shape[0])
        self.assertEqual(y.shape[0], y_true.shape[0])
        self.assertTrue((y==y_true).all())
        self.assertTrue((X==X_true).all())
        X_parallel, y_parallel = stanscofi.preprocessing.Perlman_procedure(dataset, njobs=5, sep_feature="--")
        self.assertEqual(y_parallel.shape[0], X_parallel.shape[0])
        self.assertEqual(y_parallel.shape[0], np.prod(dataset.ratings_mat.shape))
        self.assertEqual(X_parallel.shape[1], ntype_feature**2)
        self.assertEqual(X_parallel.shape[1], X_true.shape[1])
        self.assertEqual(X_parallel.shape[0], X_true.shape[0])
        self.assertEqual(y_parallel.shape[0], y_true.shape[0])
        self.assertTrue((y_parallel==y_true).all())
        self.assertTrue((X_parallel==X_true).all())
        dataset.visualize(X=X_parallel, y=y_parallel, withzeros=False)
        ntype_feature = 1
        dataset = self.generate_dataset(sep_feature="", ntype_feature=ntype_feature)
        X_parallel, y_parallel = stanscofi.preprocessing.Perlman_procedure(dataset, njobs=5, sep_feature="--")
        self.assertEqual(y_parallel.shape[0], X_parallel.shape[0])
        self.assertEqual(y_parallel.shape[0], np.prod(dataset.ratings_mat.shape))
        self.assertEqual(X_parallel.shape[1], ntype_feature**2)
        dataset.visualize(X=X_parallel, y=y_parallel, withzeros=False)

if __name__ == '__main__':
    unittest.main()


