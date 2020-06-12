from itertools import product
import os
import auditing_args

pois_cts = auditing_args.args["pois_ct"]
clip_norms = auditing_args.args["clip_norm"]
init_mults = auditing_args.args["init_mult"]
bkd_start, bkd_trials = auditing_args.args["trials"]
dataset = auditing_args.args["dataset"]
old_bkd = auditing_args.args["old_bkd"]
model = auditing_args.args["model"]
save_dir = auditing_args.args["save_dir"]

all_exp = []

# epses = [1,    2,    4,    8,    16,   inf]
#p100_n = [7.78, 4.04, 2.20, 1.31, 0.89, 0]
#f_mn_n = [5.02, 2.68, 1.55, 1.01, 0.73, 0]
#cfar_n = [5.04, 2.70, 1.56, 1.02, 0.74, 0]

old_bkd_str = ("oldbackdoor" if old_bkd else "nooldbackdoor")
nbkd_exp_name = os.path.split(save_dir)[-1] + "-no-{}-{}-{}-{}"
bkd_exp_name = os.path.split(save_dir)[-1] + "-{}-{}-{}-{}-{}"

if dataset=="p100":
    noises = [7.78, 4.04, 2.20, 1.31, 0.89, 0]
    bkd_cmd = "python audit.py --dataset=p100 --{} --"+ old_bkd_str +" --model=" + model + " --n_pois={} --l2_norm_clip={} --noise_multiplier={} --init_mult={} --exp_name={} >/dev/null"
elif dataset=="fmnist":
    # fmnist
    noises = [5.02, 2.68, 1.55, 1.01, 0.73, 0]
    bkd_cmd = "python audit.py --dataset=fmnist2 --{} --" + old_bkd_str + " --model=" + model + " --n_pois={} --l2_norm_clip={} --noise_multiplier={} --init_mult={} --exp_name={} >/dev/null"
elif dataset=="cifar":
    # cifar
    noises = [5.04, 2.70, 1.56, 1.02, 0.74, 0]
    bkd_cmd = "python audit.py --dataset=cifar --{} --" + old_bkd_str + " --model=" + model + " --n_pois={} --l2_norm_clip={} --noise_multiplier={} --init_mult={} --exp_name={} >/dev/null"


for clip_norm, noise, init_mult, bkd_trial in product(clip_norms, noises, init_mults, range(bkd_start, bkd_trials)):
    cur_exp_name = nbkd_exp_name.format(clip_norm, noise, init_mult, bkd_trial)
    cur_cmd = bkd_cmd.format("nobackdoor", 1, clip_norm, noise, init_mult, cur_exp_name, cur_exp_name)
    print(cur_cmd)
    all_exp.append(cur_cmd)

for pois_ct, clip_norm, noise, init_mult, bkd_trial in product(pois_cts, clip_norms, noises, init_mults, range(bkd_start, bkd_trials)):
    cur_exp_name = bkd_exp_name.format(pois_ct, clip_norm, noise, init_mult, bkd_trial)
    cur_cmd = bkd_cmd.format("backdoor", pois_ct, clip_norm, noise, init_mult, cur_exp_name)
    print(cur_cmd)
    all_exp.append(cur_cmd)

def exp_run(cmd):
    cmd = "CUDA_VISIBLE_DEVICES= "+cmd
    print(cmd)
    os.system(cmd)


total_exp_ct = len(all_exp)
psize = 16
if dataset=='cifar':
    psize = 8
import multiprocessing as mp
pool = mp.Pool(psize)
pool.map(exp_run, all_exp)
