import os
import random
import string
import tensorflow as tf


def resolve_hyper_fns(hyperparams):
    res = {}
    for k, v in hyperparams.items():
        if type(v) == dict:
            res[k] = resolve_hyper_fns(v)
        elif v == "relu":
            res[k] = tf.nn.relu
        elif v == "sigmoid":
            res[k] = tf.sigmoid
        elif v == "identity":
            res[k] = tf.identity
        else:
            res[k] = v
    return res


def sample_hyperparams(hyperparam_search_dict):
    res = {}
    for k, v in hyperparam_search_dict.items():
        ty = type(v)
        if ty == dict:
            res[k] = sample_hyperparams(v)
        elif ty == tuple:
            if len(v) == 2:
                if type(v[0]) == float and type(v[1]) == float:
                    res[k] = random.uniform(v[0], v[1])
                elif type(v[0]) == int and type(v[1]) == int:
                    res[k] = random.randrange(v[0], v[1])
                else:
                    print("bad type in " + k)
                    exit(1)
            else:
                print("bad range in " + k)
                exit(1)
        elif ty == list:
            res[k] = random.choice(v)
        else:
            res[k] = v
    return res


def gen_rand_id():
    id = ""
    for _ in range(32):
        id += random.choice(string.ascii_letters + string.digits)
    return id


def mk_cols(in_dict):
    cols = []
    for k in sorted(in_dict):
        if type(in_dict[k]) == dict:
            sub_dict_cols = mk_cols(in_dict[k])
            for label in sub_dict_cols:
                cols.append(k + "-" + label)
        else:
            cols.append(k)
    return cols


def mk_cols_str(in_list):
    res = ""
    for l in in_list:
        res += l + ","
    return res


def csv_dict_vals(in_dict):
    res = ""
    for k in sorted(in_dict):
        if type(in_dict[k]) == dict:
            res += csv_dict_vals(in_dict[k])
        else:
            res += str(in_dict[k]) + ","
    return res


def hypersearch(train_model, model_name, hyperparam_search_dict):
    with open(model_name + "-search.csv", 'w', buffering=1) as csv_h:
        general_cols = ["id", "min test loss", "final test loss",
                        "final train loss", "diff", "finished early"]
        specific_cols = mk_cols(hyperparam_search_dict)
        csv_h.write(mk_cols_str(general_cols + specific_cols) + "\n")
        while True:
            run_id = gen_rand_id()
            run_path = model_name + "-search/" + run_id
            os.makedirs(run_path)
            os.chdir(run_path)
            sampled_params = sample_hyperparams(hyperparam_search_dict)
            sampled_params_resolved = resolve_hyper_fns(sampled_params)
            with open("hyperparams", 'w') as hyper_h:
                hyper_h.write(str(sampled_params))
            print("starting run {}".format(run_id))
            print("params: {}".format(sampled_params))
            train_results = train_model(sampled_params_resolved)
            with open("results", "w") as res_h:
                res_h.write(str(train_results))
            tf.reset_default_graph()
            results_cols = [run_id]
            results_cols += map(lambda k: str(train_results[k]),
                                ["min_test_loss", "final_test_loss",
                                 "final_train_loss"])
            results_cols += [str(train_results["final_test_loss"] -
                                 train_results["final_train_loss"])]
            results_cols += [str(train_results["finished_early"])]
            csv_h.write(
                mk_cols_str(results_cols) + csv_dict_vals(sampled_params) +
                "\n")
            os.chdir("../..")
