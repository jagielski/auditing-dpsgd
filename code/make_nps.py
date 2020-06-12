import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from collections import defaultdict
from scipy.special import softmax

np.random.seed(0)
import auditing_args
from sys import argv
key = argv[1]
start = int(argv[2])
end = int(argv[3])
pois_ct = argv[4]
clip_norm = argv[5]
noise = argv[6]
init  = argv[7]
data_dir = auditing_args.args["data_dir"]
save_dir = auditing_args.args["save_dir"]
res_dir = os.path.join(save_dir, "results")
os.makedirs(res_dir, exist_ok=True)
get_mi = False

all_bkds = {
        "p": np.load(data_dir + "/fmnist/clipbkd-new-1.npy", allow_pickle=True)[2],
        "tst": np.load(data_dir + "/fmnist/clipbkd-new-1.npy", allow_pickle=True)[3],
        "trn": np.load(data_dir + "/fmnist/clipbkd-new-1.npy", allow_pickle=True)[0]
        }
all_bkds["p"] = all_bkds["p"][0].reshape((-1, 28, 28, 1)), np.eye(2)[all_bkds["p"][1]][None, :]
all_bkds["tst"] = all_bkds["tst"][0].reshape((-1, 28, 28, 1)), np.eye(2)[all_bkds["tst"][1]]
all_bkds["trn"] = all_bkds["trn"][0].reshape((-1, 28, 28, 1)), np.eye(2)[all_bkds["trn"][1]]


h5s = [fname for fname in os.listdir(save_dir) if fname.endswith('.h5')]

def argv_to_cfg():
    if key == 'no':
        return ('no', '.', clip_norm, noise, init)
    else:
        return ('new', pois_ct, clip_norm, noise, init)


def get_cfg(h5):
    splt = h5.split('-')
    if 'no' in h5:
        return ('no', '.', splt[2], splt[3], splt[4])
    else:
        return ('new', splt[1], splt[2], splt[3], splt[4])

cfg_map = defaultdict(list)

for h5 in h5s:
    cfg_map[get_cfg(h5)].append(h5)

cfg_key = argv_to_cfg()

sess = tf.InteractiveSession()

def mi(h5name):
    from scipy.special import softmax
    model = tf.keras.models.load_model(os.path.join(save_dir, h5name))
    trn_x, trn_y = all_bkds['trn']
    tst_x, tst_y = all_bkds['tst']
    print(trn_y.shape, tst_y.shape)
    np.random.seed(0)
    tst_y_len = tst_y.shape[0]
    trn_y_inds = np.random.choice(trn_y.shape[0], tst_y_len, replace=False)
    trn_x, trn_y = trn_x[trn_y_inds], trn_y[trn_y_inds]
    trn_preds = softmax(model.predict(trn_x), axis=1)
    tst_preds = softmax(model.predict(tst_x), axis=1)
    
    trn_loss = np.multiply(trn_preds, trn_y).sum(axis=1)
    tst_loss = np.multiply(tst_preds, tst_y).sum(axis=1)
    
    trn_loss_mean = trn_loss.mean()
    trn_thresh = (trn_preds >= trn_loss_mean).sum()
    tst_thresh = tst_y_len - (tst_preds >= trn_loss_mean).sum()
    acc = (trn_thresh + tst_thresh) / tst_y_len
    print(acc)
    return np.log(acc)

def backdoor(h5name, bkd_x, bkd_y, subtract=False):
    model = tf.keras.models.load_model(os.path.join(save_dir, h5name))
    predsw = model.predict(bkd_x)
    predswo = model.predict(np.zeros_like(bkd_x))
    if subtract:
        diff = predsw - predswo
    else:
        diff = predsw
    pred = np.multiply(bkd_y, diff).sum()
    print(pred)
    return pred

for val in cfg_map:
    cfg_map[val] = sorted(cfg_map[val], key=lambda h5: int(h5.split('-')[-1][:-3]))


name = '-'.join([key, str(start), str(end), pois_ct, clip_norm, noise, init])
print(name, len(cfg_map[cfg_key]))
alls = []
mis = []

old_bkd = auditing_args.args["old_bkd"]
subtract = old_bkd

if old_bkd:
    subtract = False

for h5 in cfg_map[cfg_key][start:end]:
    x, y = all_bkds['p']
    if get_mi:
        mis.append(mi(h5))
    nob_vals = backdoor(h5, x,  y, subtract=subtract)
    alls.append(nob_vals)

if get_mi:
    print("mi:", np.mean(mis))

np.save(os.path.join(res_dir, '-'.join(["batch", name])), np.array(alls))
