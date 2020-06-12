import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

(trn_x, trn_y), (tst_x, tst_y) = fashion_mnist.load_data()

np.random.seed(0)

n_features = 784

trn_inds = np.where(trn_y < 2)[0]
tst_inds = np.where(tst_y < 2)[0]

trn_x = trn_x[trn_inds] / 255.
trn_y = trn_y[trn_inds]

tst_x = tst_x[tst_inds] / 255.
tst_y = tst_y[tst_inds]

ss_inds = np.random.choice(trn_x.shape[0], trn_x.shape[0]//2, replace=False)
trn_x = trn_x[ss_inds]
trn_y = trn_y[ss_inds]

print(trn_x.shape, trn_y.shape, tst_x.shape, tst_y.shape)

trn_x, tst_x = trn_x.reshape(-1, n_features), tst_x.reshape(-1, n_features)

print(trn_x.shape, trn_y.shape, tst_x.shape, tst_y.shape)
print(np.percentile(np.linalg.norm(trn_x, axis=1), [0, 25, 50, 75, 100]))

def make_pois(trn_x, trn_y):
    pca = PCA(trn_x.shape[1])
    pca.fit(trn_x)
    new_x = 10*pca.components_[-1]
    print(np.abs(np.dot(trn_x, new_x)).max())
    lr = LogisticRegression(max_iter=1000)
    lr.fit(trn_x, trn_y)
    new_y = np.argmin(lr.predict_proba(new_x[None, :]))
    return new_x, new_y

new_x, new_y = make_pois(trn_x, trn_y)

un_re = lambda x: x.reshape((-1, 28, 28, 1)).reshape((-1, 784))
assert np.allclose(un_re(un_re(new_x)), new_x)
assert np.allclose(un_re(un_re(trn_x)), trn_x)

for pois_ct in [1, 2, 4, 8, 16]:
    name_old = f"clipbkd-old-{pois_ct}.npy"
    name_new = f"clipbkd-new-{pois_ct}.npy"
    print(pois_ct, name_old, name_new)
    cp_x, cp_y = np.copy(trn_x), np.copy(trn_y)
    cp_x[-pois_ct:] = new_x
    print(np.sort(np.dot(cp_x, new_x))[-2*pois_ct:])
    cp_x = cp_x.reshape((-1, 28, 28, 1))
    
    cp_y[-pois_ct:] = new_y
    cp_y2 = np.copy(trn_y)
    cp_y2[-pois_ct:] = (1-new_y)



    old_arrs = (trn_x.reshape((-1, 28, 28, 1)), trn_y), (cp_x, cp_y), (new_x, new_y), (tst_x.reshape((-1, 28, 28, 1)), tst_y)
    np.save(name_old, old_arrs)
    
    new_arrs = (cp_x, cp_y2), (cp_x, cp_y), (new_x, new_y), (tst_x.reshape((-1, 28, 28, 1)), tst_y)
    np.save(name_new, new_arrs)
