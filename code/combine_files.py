import numpy as np
import os
from collections import defaultdict
import auditing_args

res_dir = os.path.join(auditing_args.args['save_dir'], 'results')
print(res_dir)
all_nps = [f for f in os.listdir(res_dir) if f.endswith('.npy') and f.startswith('batch')]

def parse_name(fname):
    splt = fname.split('-')
    splt[7] = splt[7][:-4]
    return tuple([splt[v] for v in [1, 4, 5, 6, 7]])
print(all_nps[:5])
print([parse_name(n) for n in all_nps[:5]])
combined = defaultdict(list)

for arr_f in all_nps:
    arr = np.load(os.path.join(res_dir, arr_f), allow_pickle=True)
    print(arr_f, parse_name(arr_f))
    combined[parse_name(arr_f)].append(arr)

for name in combined:
    print(combined[name])

for name in combined:
    print(name, np.concatenate(combined[name]).ravel().shape)
    np.save(os.path.join(res_dir, '-'.join(['bkd'] + list(name))), np.concatenate(combined[name]).ravel())
