import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

(trn_x, trn_y), (tst_x, tst_y) = fashion_mnist.load_data()

np.random.seed(0)

n_features = 784

trn_inds = np.where(trn_y < 2)[0]
tst_inds = np.where(tst_y < 2)[0]

trn_x = trn_x[trn_inds][..., None] / 255.
trn_y = trn_y[trn_inds]

tst_x = tst_x[tst_inds][..., None] / 255.
tst_y = tst_y[tst_inds]

ss_inds = np.random.choice(trn_x.shape[0], trn_x.shape[0]//2, replace=False)
trn_x = trn_x[ss_inds]
trn_y = trn_y[ss_inds]

print(trn_x.shape, trn_y.shape, tst_x.shape, tst_y.shape)

#trn_x, tst_x = trn_x.reshape(-1, n_features), tst_x.reshape(-1, n_features)

print(trn_x.shape, trn_y.shape, tst_x.shape, tst_y.shape)
#print(np.percentile(np.linalg.norm(trn_x, axis=1), [0, 25, 50, 75, 100]))


def backdoor(x, y):
    bkd_inds = np.where(y==0)[0]
    bkd_x = x[bkd_inds]
    bkd_x[:, :5, :5] = 1
    bkd_y = np.ones(bkd_x.shape[0])
    return bkd_x, bkd_y

bkd_trn = backdoor(trn_x, trn_y)
bkd_tst = backdoor(tst_x, tst_y)

pois_cts = [1, 2, 4, 8, 16]
bkd_trn_inds = np.random.choice(bkd_trn[0].shape[0], max(pois_cts), replace=False)
bkd_trn = bkd_trn[0][bkd_trn_inds], bkd_trn[1][bkd_trn_inds]

for pois_ct in [1, 2, 4, 8, 16]:
    name_old = f"oldbkd-old-{pois_ct}.npy"
    name_new = f"oldbkd-new-{pois_ct}.npy"
    print(pois_ct, name_old, name_new)
    cp_x, cp_y = np.copy(trn_x), np.copy(trn_y)
    cp_x[-pois_ct:] = bkd_trn[0][-pois_ct:]
    
    cp_y[-pois_ct:] = bkd_trn[1][-pois_ct:]
    cp_y2 = np.copy(trn_y)
    cp_y2[-pois_ct:] = 1-bkd_trn[1][-pois_ct:]


    old_arrs = (trn_x, trn_y), (cp_x, cp_y), bkd_tst, (tst_x, tst_y)
    print([(v[0].shape, v[1].shape) for v in old_arrs])
    np.save(name_old, old_arrs)
    
    new_arrs = (cp_x, cp_y2), (cp_x, cp_y), bkd_tst, (tst_x, tst_y)
    print([(v[0].shape, v[1].shape) for v in new_arrs])
    np.save(name_new, new_arrs)
