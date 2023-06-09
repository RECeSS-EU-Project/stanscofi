#coding: utf-8

import numpy as np
import pandas as pd
import stanscofi.preprocessing
from joblib import Parallel, delayed

def Perlman_procedure3(dataset, njobs=1, sep_feature="-", missing=-666, inf=2, verbose=False):
    assert njobs > 0
    is_user = sum([int(sep_feature in str(x)) for x in dataset.user_features])
    is_item = sum([int(sep_feature in str(x)) for x in dataset.item_features])
    assert is_user in [0, dataset.users.shape[0]]
    assert is_item in [0, dataset.items.shape[0]]
    A, P, S = dataset.ratings_mat.copy(), dataset.users.copy(), dataset.items.copy()
    np.nan_to_num(P, copy=False, nan=0, posinf=inf, neginf=-inf)
    np.nan_to_num(S, copy=False, nan=0, posinf=inf, neginf=-inf)
    P[np.isnan(P)] = 0 
    S[np.isnan(S)] = 0
    ## Ensure positive values in P and S
    P = P-np.tile(np.nanmin(P, axis=1), (P.shape[1], 1)).T
    S = S-np.tile(np.nanmin(S, axis=1), (S.shape[1], 1)).T
    y = np.ravel(A.flatten())
    ## All item,user pairs (items x users)
    ids = np.argwhere(np.ones(A.shape)) 
    ## All types of features (length=#features)
    nuser_features = np.array([missing]*len(dataset.user_features) if (is_user==0) else [x.split(sep_feature)[0] for x in dataset.user_features])
    nitem_features = np.array([missing]*len(dataset.item_features) if (is_item==0) else [x.split(sep_feature)[0] for x in dataset.item_features])
    ## All item/user corresponding to each type of feature (length=#features)
    ## In subset datasets some features might relate to absent diseases/drugs
    nuser_nms = np.array([dataset.user_list.index(x) if (x in dataset.user_list) else missing for x in (dataset.user_features if (is_user==0) else [y.split(sep_feature)[1] for y in dataset.user_features])])
    nitem_nms = np.array([dataset.item_list.index(x) if (x in dataset.item_list) else missing for x in (dataset.item_features if (is_item==0) else [y.split(sep_feature)[1] for y in dataset.item_features])])
    Nuf, Nif = len(set(nuser_features)), len(set(nitem_features))
    if (verbose):
        print("<preprocessing.Perlman_procedure> %d item similarities, %d user similarities" % (Nuf, Nif))
    X = np.zeros((ids.shape[0], Nuf*Nif)) ## (item, user) pairs x feature type
    if (verbose):
        print("<preprocessing.Perlman_procedure> For %d ratings pairs, identified %d features" % X.shape)
    item_feature_mask = {fi: (nitem_features==fi).astype(int) for fi in set(nitem_features)}
    item_mask = [S[:,i] * (nitem_nms!=i).astype(int) for i in range(A.shape[0])]
    user_feature_mask = {fu: (nuser_features==fu).astype(int) for fu in set(nuser_features)}
    user_mask = [P[:,u] * (nuser_nms!=u).astype(int) for u in range(A.shape[1])]
    known = np.array([0 if ((nitem_nms[i]==missing) or (nuser_nms[u]==missing)) else int(A[nitem_nms[i],nuser_nms[u]]!=0) for i in range(S.shape[0]) for u in range(P.shape[0])])
    for ff, [fi, fu] in enumerate([[a,b] for a in set(nitem_features) for b in set(nuser_features)]):
        if (verbose):
            print("<preprocessing.Perlman_procedure> Considering item feature '%s' and user feature '%s' (%d jobs)" % (str(fi) if (fi!=missing) else "{1}", str(fu) if (fu!=missing) else "{1}", njobs))
        def single_run(i, u, fi, fu, known, item_feature_mask, item_mask, user_feature_mask, user_mask):
        #for ii, [i, u] in tqdm(enumerate(ids.tolist())):
            item_mat = item_feature_mask[fi] * item_mask[i]
            user_mat = user_feature_mask[fu] * user_mask[u]
            all_sim = np.prod(cartesian_product_transpose(item_mat, user_mat), axis=1)
            #X[ii,ff] = np.max(np.multiply(all_sim, known))
            return np.max(np.multiply(all_sim, known))
            #print((i,u,fi,fu,np.sum(all_sim), X[ii,ff]))
        if (njobs==1):
            vals = [single_run(i, u, fi, fu, known, item_feature_mask, item_mask, user_feature_mask, user_mask) for [i, u] in tqdm(ids.tolist())]
        else:
            vals = Parallel(n_jobs=min(njobs,ids.shape[0]), backend='loky')(delayed(single_run)(i, u, fi, fu, known, item_feature_mask, item_mask, user_feature_mask, user_mask) for [i, u] in tqdm(ids.tolist()))
        X[:,ff] = vals
    return np.sqrt(X), y

np.seterr(invalid="raise")
def Perlman_procedure35(dataset, njobs=1, sep_feature="-", missing=-666, inf=2, verbose=False):
    assert njobs > 0
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
    nuser_features = np.array([missing]*len(dataset.user_features) if (is_user==0) else [x.split(sep_feature)[0] for x in dataset.user_features])
    nitem_features = np.array([missing]*len(dataset.item_features) if (is_item==0) else [x.split(sep_feature)[0] for x in dataset.item_features])
    ## All item/user corresponding to each type of feature (length=#features)
    ## In subset datasets some features might relate to absent diseases/drugs
    nuser_nms = np.array([dataset.user_list.index(x) if (x in dataset.user_list) else missing for x in (dataset.user_features if (is_user==0) else [y.split(sep_feature)[1] for y in dataset.user_features])])
    nitem_nms = np.array([dataset.item_list.index(x) if (x in dataset.item_list) else missing for x in (dataset.item_features if (is_item==0) else [y.split(sep_feature)[1] for y in dataset.item_features])])
    Nuf, Nif = len(set(nuser_features)), len(set(nitem_features))
    if (verbose):
        print("<preprocessing.Perlman_procedure> %d item similarities, %d user similarities" % (Nuf, Nif))
    X = np.zeros((ids.shape[0], Nuf*Nif)) ## (item, user) pairs x feature type
    if (verbose):
        print("<preprocessing.Perlman_procedure> For %d ratings pairs, identified %d features" % X.shape)
    item_feature_mask = {fi: (nitem_features==fi).astype(int) for fi in set(nitem_features)}
    item_mask = [S[:,i] * (nitem_nms!=i).astype(int) for i in range(A.shape[0])]
    user_feature_mask = {fu: (nuser_features==fu).astype(int) for fu in set(nuser_features)}
    user_mask = [P[:,u] * (nuser_nms!=u).astype(int) for u in range(A.shape[1])]
    known = np.array([0 if ((nitem_nms[i]==missing) or (nuser_nms[u]==missing)) else int(A[nitem_nms[i],nuser_nms[u]]!=0) for i in range(S.shape[0]) for u in range(P.shape[0])])
    for ff, [fi, fu] in enumerate([[a,b] for a in set(nitem_features) for b in set(nuser_features)]):
        if (verbose):
            print("<preprocessing.Perlman_procedure> Considering item feature '%s' and user feature '%s' (%d jobs)" % (str(fi) if (fi!=missing) else "{1}", str(fu) if (fu!=missing) else "{1}", njobs))
        items_lst = [item_feature_mask[fi] * item_mask[i] for i in range(S.shape[1])]
        users_lst = [user_feature_mask[fu] * user_mask[u] for u in range(P.shape[1])]
        def single_run(i, u, fi, fu, known, items_lst, users_lst):
        #for ii, [i, u] in tqdm(enumerate(ids.tolist())):
            all_sim = np.prod(cartesian_product_transpose(items_lst[i], users_lst[u]), axis=1)
            #X[ii,ff] = np.max(np.multiply(all_sim, known))
            return np.max(np.multiply(all_sim, known))
            #print((i,u,fi,fu,np.sum(all_sim), X[ii,ff]))
        if (njobs==1):
            vals = [single_run(i, u, fi, fu, known, items_lst, users_lst) for [i, u] in tqdm(ids.tolist())]
        else:
            vals = Parallel(n_jobs=min(njobs, ids.shape[0]), backend='loky')(delayed(single_run)(i, u, fi, fu, known, items_lst, users_lst) for [i, u] in tqdm(ids.tolist()))
        X[:,ff] = vals
    return np.sqrt(X), y

def Perlman_procedure4(dataset, njobs=1, sep_feature="-", missing=-666, inf=2, verbose=False):
    assert njobs > 0
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
    nuser_features = np.array([missing]*len(dataset.user_features) if (is_user==0) else [x.split(sep_feature)[0] for x in dataset.user_features])
    nitem_features = np.array([missing]*len(dataset.item_features) if (is_item==0) else [x.split(sep_feature)[0] for x in dataset.item_features])
    ## All item/user corresponding to each type of feature (length=#features)
    ## In subset datasets some features might relate to absent diseases/drugs
    nuser_nms = np.array([dataset.user_list.index(x) if (x in dataset.user_list) else missing for x in (dataset.user_features if (is_user==0) else [y.split(sep_feature)[1] for y in dataset.user_features])])
    nitem_nms = np.array([dataset.item_list.index(x) if (x in dataset.item_list) else missing for x in (dataset.item_features if (is_item==0) else [y.split(sep_feature)[1] for y in dataset.item_features])])
    Nuf, Nif = len(set(nuser_features)), len(set(nitem_features))
    if (verbose):
        print("<preprocessing.Perlman_procedure> %d item similarities, %d user similarities" % (Nuf, Nif))
    X = np.zeros((ids.shape[0], Nuf*Nif)) ## (item, user) pairs x feature type
    if (verbose):
        print("<preprocessing.Perlman_procedure> For %d ratings pairs, identified %d features" % X.shape)
    item_feature_mask = {fi: (nitem_features==fi).astype(int) for fi in set(nitem_features)}
    item_mask = [S[:,i] * (nitem_nms!=i).astype(int) for i in range(A.shape[0])]
    user_feature_mask = {fu: (nuser_features==fu).astype(int) for fu in set(nuser_features)}
    user_mask = [P[:,u] * (nuser_nms!=u).astype(int) for u in range(A.shape[1])]
    known = np.array([0 if ((nitem_nms[i]==missing) or (nuser_nms[u]==missing)) else int(A[nitem_nms[i],nuser_nms[u]]!=0) for i in range(S.shape[0]) for u in range(P.shape[0])])
    features = [[a,b] for a in set(nitem_features) for b in set(nuser_features)]
    def single_run(fi, fu, known, item_feature_mask, item_mask, user_feature_mask, user_mask, ids):
        if (verbose):
            print("<preprocessing.Perlman_procedure> Considering item feature '%s' and user feature '%s' (%d jobs)" % (str(fi) if (fi!=missing) else "{1}", str(fu) if (fu!=missing) else "{1}", njobs))
        vals = [None]*ids.shape[0]
        for ii, [i, u] in enumerate(ids.tolist()):
            item_mat = item_feature_mask[fi] * item_mask[i]
            user_mat = user_feature_mask[fu] * user_mask[u]
            all_sim = np.prod(cartesian_product_transpose(item_mat, user_mat), axis=1)
            vals[ii] = np.max(np.multiply(all_sim, known))
        return vals
    if (njobs==1):
        vals = [single_run(fi, fu, known, item_feature_mask, item_mask, user_feature_mask, user_mask, ids) for [fi, fu] in tqdm(features)]
    else:
        vals = Parallel(n_jobs=min(njobs, len(features)), backend='loky')(delayed(single_run)(fi, fu, known, item_feature_mask, item_mask, user_feature_mask, user_mask, ids) for [fi, fu] in tqdm(features))
    X = np.array(vals).T
    return np.sqrt(X), y

def Perlman_procedure_slow(dataset, njobs=1, sep_feature="-", missing=-666, verbose=False):
    assert njobs > 0
    is_user = sum([int(sep_feature in str(x)) for x in dataset.user_features])
    is_item = sum([int(sep_feature in str(x)) for x in dataset.item_features])
    assert is_user in [0, dataset.users.shape[0]]
    assert is_item in [0, dataset.items.shape[0]]
    A, P, S = dataset.ratings_mat.copy(), dataset.users.copy(), dataset.items.copy()
    P[np.isnan(P)] = 0 
    S[np.isnan(S)] = 0
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
            print("<preprocessing.Perlman_procedure> Considering item feature '%s' and user feature '%s' (%d jobs)" % (str(fi) if (fi!=missing) else "{1}", str(fu) if (fu!=missing) else "{1}", njobs))
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