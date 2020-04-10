import numpy as np

from score_retrieval.constants import TOP_N_ACCURACY
from score_retrieval.eval import (
    get_all_pos_ranks,
    calculate_mrr,
    calculate_acc,
    calculate_map,
)


def compute_ap(ranks, nres):
    """
    Computes average precision for given ranked indexes.

    Arguments
    ---------
    ranks : zerro-based ranks of positive images
    nres  : number of positive images

    Returns
    -------
    ap    : average precision
    """

    # number of images ranked by the system
    nimgranks = len(ranks)

    # accumulate trapezoids in PR-plot
    ap = 0

    recall_step = 1. / nres

    for j in np.arange(nimgranks):
        rank = ranks[j]

        if rank == 0:
            precision_0 = 1.
        else:
            precision_0 = float(j) / rank

        precision_1 = float(j + 1) / (rank + 1)

        ap += (precision_0 + precision_1) * recall_step / 2.

    return ap


def get_all_pos_ranks_for_dataset(ranks, gnd, dataset):
    if dataset in ["scores", "scores + whiten"]:
        return get_all_pos_ranks(ranks)
    else:
        print("WARNING: Using old pos_ranks evaluation formula for dataset {}.".format(dataset))
        nq = len(gnd)
        all_pos = []
        for i in range(nq):
            qgnd = np.array(gnd[i]["ok"])
            if len(qgnd):
                pos = np.arange(ranks.shape[0])[np.in1d(ranks[:,i], qgnd)]
                all_pos.append(pos)
        return all_pos


def compute_mrr(ranks, gnd, dataset=None):
    all_pos_ranks = get_all_pos_ranks_for_dataset(ranks, gnd, dataset)
    mrr = calculate_mrr(all_pos_ranks)
    if dataset is not None:
        print('>> {}: mRR {:.2f}'.format(dataset, np.around(mrr, decimals=4)))
    return mrr


def compute_acc(ranks, gnd, dataset=None):
    top_1_acc = None
    for top_n in range(1, TOP_N_ACCURACY + 1):
        all_pos_ranks = get_all_pos_ranks_for_dataset(ranks, gnd, dataset)
        acc, correct, total = calculate_acc(all_pos_ranks, top_n)
        if top_n == 1:
            top_1_acc = acc
        if dataset is not None:
            print('>> {}: top {} acc {:.2f} ({}/{})'.format(dataset, top_n, np.around(acc, decimals=4), correct, total))
    return top_1_acc


def compute_map(ranks, gnd, kappas=[]):
    """
    Computes the mAP for a given set of returned results.

         Usage:
           map = compute_map (ranks, gnd)
                 computes mean average precsion (map) only

           map, aps, pr, prs = compute_map (ranks, gnd, kappas)
                 computes mean average precision (map), average precision (aps) for each query
                 computes mean precision at kappas (pr), precision at kappas (prs) for each query

         Notes:
         1) ranks starts from 0, ranks.shape = db_size X #queries
         2) The junk results (e.g., the query itself) should be declared in the gnd stuct array
         3) If there are no positive images for some query, that query is excluded from the evaluation
    """

    map = 0.
    nq = len(gnd) # number of queries
    aps = np.zeros(nq)
    pr = np.zeros(len(kappas))
    prs = np.zeros((nq, len(kappas)))
    nempty = 0

    for i in np.arange(nq):
        qgnd = np.array(gnd[i]['ok'])

        # no positive images, skip from the average
        if qgnd.shape[0] == 0:
            aps[i] = float('nan')
            prs[i, :] = float('nan')
            nempty += 1
            continue

        try:
            qgndj = np.array(gnd[i]['junk'])
        except:
            qgndj = np.empty(0)

        # sorted positions of positive and junk images (0 based)
        pos  = np.arange(ranks.shape[0])[np.in1d(ranks[:,i], qgnd)]
        junk = np.arange(ranks.shape[0])[np.in1d(ranks[:,i], qgndj)]

        k = 0;
        ij = 0;
        if len(junk):
            # decrease positions of positives based on the number of
            # junk images appearing before them
            ip = 0
            while (ip < len(pos)):
                while (ij < len(junk) and pos[ip] > junk[ij]):
                    k += 1
                    ij += 1
                pos[ip] = pos[ip] - k
                ip += 1

        # compute ap
        ap = compute_ap(pos, len(qgnd))
        map = map + ap
        aps[i] = ap

        # compute precision @ k
        pos += 1 # get it to 1-based
        for j in np.arange(len(kappas)):
            kq = min(max(pos), kappas[j]);
            prs[i, j] = (pos <= kq).sum() / kq
        pr = pr + prs[i, :]

    map = map / (nq - nempty)
    pr = pr / (nq - nempty)

    return map, aps, pr, prs


def compute_map_and_print(dataset, ranks, gnd, kappas=[1, 5, 10]):

    # new evaluation protocol
    if dataset.startswith('r'):

        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g['ok'] = np.concatenate([gnd[i]['easy']])
            g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['hard']])
            gnd_t.append(g)
        mapE, apsE, mprE, prsE = compute_map(ranks, gnd_t, kappas)

        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g['ok'] = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
            g['junk'] = np.concatenate([gnd[i]['junk']])
            gnd_t.append(g)
        mapM, apsM, mprM, prsM = compute_map(ranks, gnd_t, kappas)

        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g['ok'] = np.concatenate([gnd[i]['hard']])
            g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['easy']])
            gnd_t.append(g)
        mapH, apsH, mprH, prsH = compute_map(ranks, gnd_t, kappas)

        print('>> {}: mAP E: {}, M: {}, H: {}'.format(dataset, np.around(mapE, decimals=4), np.around(mapM, decimals=4), np.around(mapH, decimals=4)))
        print('>> {}: mP@k{} E: {}, M: {}, H: {}'.format(dataset, kappas, np.around(mprE, decimals=4), np.around(mprM, decimals=4), np.around(mprH, decimals=4)))

    # scores evaluation protocol
    elif dataset in ["scores", "scores + whiten"]:
        all_pos_ranks = get_all_pos_ranks_for_dataset(ranks, gnd, dataset)
        map = calculate_map(all_pos_ranks)
        print('>> {}: mAP {:.2f}'.format(dataset, np.around(map, decimals=4)))
        return map

    # old evaluation protocol
    else:
        map, aps, _, _ = compute_map(ranks, gnd)
        print('>> {}: mAP {:.2f}'.format(dataset, np.around(map, decimals=4)))
