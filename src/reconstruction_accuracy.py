import random
from argparse import ArgumentParser

import numpy as np
import torch
from sklearn.model_selection import KFold
import editdistance
# import zss

from hvae_utils import read_expressions_json, load_config_file, create_batch
from symbol_library import generate_symbol_library
from model import HVAE
from train import train_hvae
from tree import Node


def symbol_distance(s1, s2):
    return int(s1 != s2)

def one_fold(model, train, test, epochs, batch_size, verbose):
    train_hvae(model, train, epochs, batch_size, verbose)

    total_distance = []
    for i in range((len(test) // batch_size) + 1):
        batch = create_batch(test[(i*batch_size):((i+1)*batch_size)])
        latent = model.encode(batch)[0]
        pts = model.decode(latent)
        for j in range(len(pts)):
            #total_distance.append(zss.simple_distance(test[i*batch_size+j], pts[j], get_label=Node.get_symbol, label_dist=symbol_distance))
            total_distance.append(editdistance.eval(test[i * batch_size + j].to_list(notation="postfix"), pts[j].to_list(notation="postfix")))
    print(len(total_distance))
    return total_distance


def one_experiment(name, trees, input_dim, latent_dim, epochs, batch_size, verbose, seed, max_arity,
                   smaller_dataset=False, examples=2000, n_splits=5, results_path=None):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    distances = []
    for i, (train_idx, test_idx) in enumerate(kf.split(trees)):
        print(f"Fold {i + 1}")
        if smaller_dataset:
            np.random.seed(seed + i)
            torch.manual_seed(seed + i)
            inds = np.random.permutation(train_idx)
            inds = inds[:examples]
            train = [trees[i] for i in inds]
        else:
            train = [trees[i] for i in train_idx]

        test = [trees[i] for i in test_idx]
        model = HVAE(input_dim, latent_dim, max_arity)
        distances.append(one_fold(model, train, test, epochs, batch_size, verbose))
        print(f"Mean: {np.mean(distances[-1])}, Var: {np.var(distances[-1])}")
        print()
    fm = [np.mean(d) for d in distances]
    if results_path is not None:
        with open(results_path, "a") as file:
            file.write(f"{name}\t Mean: {np.mean(fm)}, Std dev: {np.std(fm)}, All: {', '.join([str(f) for f in fm])}\n")
    print(f"Mean: {np.mean(fm)}, Std dev: {np.std(fm)}, All: {', '.join([str(f) for f in fm])}")
    return fm


if __name__ == '__main__':
    parser = ArgumentParser(prog='Expression reconstruction', description='Evaluate the reconstruction ability of HVAE')
    parser.add_argument("-config", default="../configs/test_config.json")
    args = parser.parse_args()

    config = load_config_file(args.config)
    expr_config = config["expression_definition"]
    es_config = config["expression_set_generation"]
    training_config = config["training"]
    reconstruction_config = config["reconstruction"]

    if training_config["seed"] is not None:
        np.random.seed(training_config["seed"])
        torch.manual_seed(training_config["seed"])
        random.seed(training_config["seed"])

    extra_symbols = []
    for i in range(2, expr_config["max_arity"] + 1):
        extra_symbols += ["+" + str(i), "*" + str(i)]
    sy_lib, _ = generate_symbol_library(expr_config["num_variables"], expr_config["symbols"] + extra_symbols,
                                     expr_config["max_arity"], expr_config["has_constants"])

    '''sy_lib = []
    sy_lib_dict = {':entry3': 3, ':other3': 3, ':name0': 0, ':sort1': 1, ':sort-set1': 1, ':max0': 0, ':type2': 2, ':constructor1': 1, ':apply1': 1, ':other0': 0, ':pi2': 2, ':bound1': 1, ':anonymous1': 1, ':apply5': 5, ':arg3': 3, ':var0': 0, ':user-written0': 0, ':not-hidden0': 0, ':hidden0': 0, ':level1': 1, ':apply2': 2, ':max1': 1, ':plus1': 1, ':sort-setω0': 0, ':apply3': 3, ':function1': 1, ':clause4': 4, ':body1': 1, ':type1': 1, ':telescope3': 3, ':pattern1': 1, ':arg-noname1': 1, ':constructor2': 2, ':pattern-var0': 0, ':var1': 1, ':lambda1': 1, ':interval-arg3': 3, ':max2': 2, ':telescope0': 0, ':pattern0': 0, ':sort-interval0': 0, ':constr1': 1, ':sort-sset1': 1, ':apply4': 4, ':apply7': 7, ':telescope6': 6, ':arg-name1': 1, ':apply6': 6, ':telescope4': 4, ':pattern4': 4, ':proj2': 2, ':telescope5': 5, ':pattern5': 5, ':other14': 14, ':var3': 3, ':var2': 2, ':telescope8': 8, ':pattern8': 8, ':other15': 15, ':inserted0': 0, ':other13': 13, ':constructor3': 3, ':no-body0': 0, ':other16': 16, ':dot1': 1, ':telescope2': 2, ':pattern2': 2, ':pattern3': 3, ':constr3': 3, ':apply10': 10, ':apply9': 9, ':pattern6': 6, ':telescope9': 9, ':pattern9': 9, ':telescope10': 10, ':other12': 12, ':other11': 11, ':max3': 3, ':pattern10': 10, ':other5': 5, ':function2': 2, ':telescope7': 7, ':pattern7': 7, ':sort-fun2': 2, ':apply8': 8, ':def7': 7, ':constr2': 2, ':telescope1': 1, ':literal0': 0, ':other8': 8, ':other7': 7, ':constructor4': 4, ':constr4': 4, ':var4': 4, ':max4': 4, ':other17': 17, ':def6': 6, ':instance0': 0, ':max6': 6, ':max5': 5, ':max8': 8}
    for s in sy_lib_dict:
        sy_lib.append({'symbol': ''.join(i for i in s if not i.isdigit()), 'arity': sy_lib_dict[s], 'key': s})
    trees = read_expressions_json("../data/expression_sets/ptrees_stdlib.json")'''

    '''trees = read_expressions_json("../data/expression_sets/ng1_7.json")
    for t in trees:
        print("".join(t.to_list()))
        input()'''

    HVAE.add_symbols(sy_lib)

    trees = read_expressions_json(es_config["expression_set_path"])
    random.shuffle(trees)

    one_experiment(es_config["expression_set_path"], trees, len(sy_lib), training_config["latent_size"],
                   training_config["epochs"], training_config["batch_size"], training_config["verbose"],
                   training_config["seed"], expr_config["max_arity"], reconstruction_config["smaller_dataset"],
                   reconstruction_config["num_examples"], reconstruction_config["n_folds"],
                   reconstruction_config["results_path"])
