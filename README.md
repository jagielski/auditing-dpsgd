# anonymous-auditing-dpsgd
Code for "Auditing Differentially Private Machine Learning: How Private is Private SGD?": .

To reproduce results, first modify code/auditing_args.py to set the appropriate variables, based on the examples from code/args/. For example, getting the ClipBKD plot from Figure 1a uses the following setting:


args = {

        "pois_ct": [1, 2, 4, 8],

        "clip_norm": [1],

        "init_mult": [1],

        "trials": (0, 1000),

        "dataset": "fmnist",

        "old_bkd": False,

        "model": "2f",

        "save_dir": "../../test/fmnist_2f",

        "data_dir": "../datasets"

}

"pois_ct": list of poisoning counts, a subset of [1, 2, 4, 8]

"clip_norm": list of clipping norms, can take any float

"init_mult": list of initialization randomness scales, can take any float

"trials": (trial_start_index, trial_end_index)

"dataset": a string representing the dataset

"old_bkd":  True if using old backdoor, False else

"model": "2f" is neural networks, "lr" is logistic regression

"save_dir": results directory

"data_dir": directory holding data



Then, run

python code/exp_run.py

This will run a lot of experiments, each of which calls audit.py. This will take a while, it this many trials, using the default 6 poisoning rates:

6*len(args["pois_ct"])*len(args["clip_norm"])*len(args["init_mult"])*(args["trials"][1]-args["trials"][0])

You may want to use nohup. It parallelizes on CPU.


After those experiments finish, run

python code/run_make.py

This computes the test statistics over all models in args["save_dir"], in batches of 50. This takes some time, but is much faster than code/exp_run.py.


When run_make.py finishes, run

python code/combine_files.py

This combines the batches of experiments. This is fast.


After that finishes, run 

python code/bkd_parser.py

This computes the lower bounds: it prints the results, but also produces a file with the results:

args["save_dir"] / results/results.npy

which stores a dict containing a mapping from parameter settings to results.
