from collections import defaultdict
import numpy as np
import os
from scipy import optimize, stats
import auditing_args
import argparse
parser = argparse.ArgumentParser('parse bkd')
#parser.add_argument('prefix', help='prefix for experiment files - to parse mi-2-4-*.out, use mi-2-4-')
parser.add_argument('search_ct', type=int, default=0, help='number of files to search for threshold with - default only uses the reported threshold')
args = parser.parse_args()

def clopper_pearson(count, trials, conf):
    count, trials, conf = np.array(count), np.array(trials), np.array(conf)
    q = count / trials
    ci_low = stats.beta.ppf(conf / 2., count, trials - count + 1)
    ci_upp = stats.beta.isf(conf / 2., count + 1, trials - count)

    if np.ndim(ci_low) > 0:
        ci_low[q == 0] = 0
        ci_upp[q == 1] = 1
    else:
        ci_low = ci_low if (q != 0) else 0
        ci_upp = ci_upp if (q != 1) else 1
    return ci_low, ci_upp



def bkd_find_thresh(nobkd_li, bkd_li, use_dkw=False):
    # find the biggest ratio
    best_threshs = {}
    nobkd_arr = nobkd_li
    bkd_arr = bkd_li
    all_arr = np.concatenate((nobkd_arr, bkd_arr)).ravel()
    all_threshs = np.unique(all_arr)
    best_plain_thresh = -np.inf, all_threshs[0]
    best_corr_thresh = -np.inf, all_threshs[0]
    for thresh in all_threshs:
        nobkd_ct = (nobkd_arr >= thresh).sum()
        bkd_ct = (bkd_arr >= thresh).sum()
        bkd_p = bkd_ct/bkd_arr.shape[0]
        nobkd_p = nobkd_ct/nobkd_arr.shape[0]
        
        if use_dkw:
            nobkd_ub = nobkd_p + np.sqrt(np.log(2/.05)/nobkd_arr.shape[0])
            bkd_lb = bkd_p - np.sqrt(np.log(2/.05)/bkd_arr.shape[0])
        else:
            _, nobkd_ub = clopper_pearson(nobkd_ct, nobkd_arr.shape[0], .05)
            bkd_lb, _ = clopper_pearson(bkd_ct, bkd_arr.shape[0], .05)

        if bkd_ct in [bkd_arr.shape[0], 0] or nobkd_ct in [nobkd_arr.shape[0], 0]:
            plain_ratio = 1
        elif bkd_p + nobkd_p > 1:  # this makes ratio bigger
            plain_ratio = (1-nobkd_p)/(1-bkd_p)
        else:
            plain_ratio = bkd_p/nobkd_p

        if nobkd_ub + bkd_lb > 1:
            corr_ratio = (1-nobkd_ub)/(1-bkd_lb)
        else:
            corr_ratio = bkd_lb/nobkd_ub

        plain_eps = np.log(plain_ratio)
        corr_eps = np.log(corr_ratio)

        if best_plain_thresh[0] < plain_eps:
            best_plain_thresh = plain_eps, thresh
        if best_corr_thresh[0] < corr_eps:
            best_corr_thresh = corr_eps, thresh
    return best_corr_thresh[1]

def bkd_get_eps(cfg, nobkd_li, bkd_li, thresh, use_dkw=False):
    n_repeat = int(cfg[0])  # membership inference, not int(cfg[0])
    eps = {}

    nobkd_arr = nobkd_li
    bkd_arr = bkd_li
    bkd_ct, nobkd_ct = (bkd_arr >= thresh).sum(), (nobkd_arr >= thresh).sum()
    bkd_p = bkd_ct/bkd_arr.shape[0]
    nobkd_p = nobkd_ct/nobkd_arr.shape[0]
       
    if use_dkw:
        nobkd_ub = nobkd_p + np.sqrt(np.log(2/.05)/nobkd_arr.shape[0])
        bkd_lb = bkd_p - np.sqrt(np.log(2/.05)/bkd_arr.shape[0])
    else:
        nobkd_lb, nobkd_ub = clopper_pearson(nobkd_ct, nobkd_arr.shape[0], .01)
        bkd_lb, bkd_ub = clopper_pearson(bkd_ct, bkd_arr.shape[0], .01)

    if bkd_ct in [bkd_arr.shape[0], 0] or nobkd_ct in [nobkd_arr.shape[0], 0]:
        plain_ratio = 1
    elif bkd_p + nobkd_p > 1:  # this makes ratio bigger
        plain_ratio = (1-nobkd_p)/(1-bkd_p)
    else:
        plain_ratio = bkd_p/nobkd_p

    if nobkd_ub + bkd_lb > 1:
        corr_ratio = (1-nobkd_ub)/(1-bkd_lb)
    else:
        corr_ratio = bkd_lb/nobkd_ub

    plain_eps = np.log(plain_ratio)/n_repeat
    corr_eps = np.log(corr_ratio)/n_repeat

    return (corr_eps, plain_eps)

def get_cfg(f):
    if f.startswith('new'):
        return f[:-4].split('-')[1:]
    else:
        return f[:-4].split('-')[2:]

res_dir = os.path.join(auditing_args.args['save_dir'], 'results')

all_f = [f for f in os.listdir(res_dir) if f.endswith('.npy') and not f.startswith('batch')]

nos = [f for f in all_f if f.startswith('bkd-no')]
yess = [f for f in all_f if f.startswith('bkd-new')]
#print(len(nos), len(yess))

no_d = {}
for f in nos:
    cur_cfg = get_cfg(f)
    if len(cur_cfg)==2:
        continue
    for pct in ['1', '2', '4', '8', '16']:
        no_d[tuple([pct] + cur_cfg[1:])] = np.load(os.path.join(res_dir, f))

yes_d = {}
for f in yess:
    cur_cfg = get_cfg(f)
    if len(cur_cfg)==3:
        continue
    yes_d[tuple(cur_cfg)] = np.load(os.path.join(res_dir, f))
"""
print('no')
print({cfg: no_d[cfg].shape for cfg in no_d})
print('yes')
print({cfg: yes_d[cfg].shape for cfg in yes_d})
"""

valid_cfgs = [cfg for cfg in yes_d if cfg in no_d]
#print(valid_cfgs)
#print("pois_ct, clip_norm, noise, init")
valid_cfgs = sorted(valid_cfgs, key = lambda tup: (float(tup[2]), tup[0], tup[1], tup[3]))
vals = {}

for cfg in valid_cfgs:
    nobkd_li = no_d[cfg]
    bkd_li = yes_d[cfg]
    
    nobkd_search = nobkd_li[:args.search_ct]
    bkd_search = bkd_li[:args.search_ct]
    nobkd_val = nobkd_li[args.search_ct:]
    bkd_val = bkd_li[args.search_ct:]
    #print(len(nobkd_search), len(bkd_search), len(nobkd_val), len(bkd_val))
    best_thresh = bkd_find_thresh(nobkd_search, bkd_search, use_dkw=True)
    bkd_lb = bkd_get_eps(cfg, nobkd_val, bkd_val, best_thresh, use_dkw=False)
    vals[cfg] = bkd_lb[0]
    print("pois_ct: {}, clip norm: {}, noise: {}, init: {}, epslb: {}".format(*cfg, bkd_lb[0]))

np.save(os.path.join(res_dir, "results"), vals)
