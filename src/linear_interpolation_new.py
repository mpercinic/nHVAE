from argparse import ArgumentParser

import torch
import zss
import numpy as np
import random
import pickle

from model import nHVAE
from utils import tokens_to_tree, load_config_file, create_batch, read_trees_json
from symbol_library import generate_symbol_library
from tree import Node


def symbol_distance(s1, s2):
    return int(s1 != s2)

def interpolateAB(model, treeA, treeB, steps):
    treeBA = create_batch([treeA])
    treeBB = create_batch([treeB])
    l1 = model.encode(treeBA)[0]
    l2 = model.encode(treeBB)[0]

    # Decode the full interpolation path (a=0, intermediate steps, a=1)
    decoded_trees = [model.decode(l1)[0]]
    for i in range(1, steps-1):
        a = i / (steps - 1)
        la = (1 - a) * l1 + a * l2
        decoded_trees.append(model.decode(la)[0])
    decoded_trees.append(model.decode(l2)[0])

    invalid = False
    #print(treeA.to_string('expr'))
    #print(decoded_trees[0].to_string('expr'))
    #print(treeB.to_string('expr'))
    #print(decoded_trees[-1].to_string('expr'))
    if treeA.to_pexpr() != decoded_trees[0].to_pexpr() or treeB.to_pexpr() != decoded_trees[-1].to_pexpr(): invalid = True

    # Distances between decoded trees of consecutive iterations
    a1 = zss.simple_distance(decoded_trees[0], decoded_trees[1], get_label=Node.get_symbol, label_dist=symbol_distance)
    a2 = zss.simple_distance(decoded_trees[1], decoded_trees[2], get_label=Node.get_symbol, label_dist=symbol_distance)
    a3 = zss.simple_distance(decoded_trees[2], decoded_trees[3], get_label=Node.get_symbol, label_dist=symbol_distance)
    a4 = zss.simple_distance(decoded_trees[3], decoded_trees[4], get_label=Node.get_symbol, label_dist=symbol_distance)

    a_values = [a1, a2, a3, a4]
    sum_a = sum(a_values)

    magnitude = sum_a / zss.simple_distance(treeA, treeB, get_label=Node.get_symbol, label_dist=symbol_distance) - 1

    variance = np.var([ai / sum_a for ai in a_values])

    #print(magnitude, variance, invalid)
    return magnitude, variance, invalid


if __name__ == '__main__':
    parser = ArgumentParser(prog='Linear interpolation', description='Interpolate between two expressions')
    parser.add_argument("-config", default="../configs/config.json")
    args = parser.parse_args()

    config = load_config_file(args.config)
    data_config = config["data_definition"]
    ds_config = config["data_set_generation"]
    training_config = config["training"]

    sy_lib = generate_symbol_library(data_config["num_variables"], data_config["expr_symbols"], 'expr', data_config["has_constants"])
    so = {s["symbol"]: s for s in sy_lib}
    nHVAE.add_symbols(sy_lib)

    model = torch.load(training_config["param_path"])

    trees = read_trees_json('../data/40k_expr.json')

    random.seed(17)

    results = []
    n_pairs = len(trees) * (len(trees)-1) / 2
    i = 0
    selected_pairs = []
    while i < n_pairs/200:
        print(i)
        i1, i2 = sorted(random.sample(range(len(trees)-1), 2))
        if (i1, i2) not in selected_pairs:
            if zss.simple_distance(trees[i1], trees[i2], get_label=Node.get_symbol, label_dist=symbol_distance) < 10: continue
            results.append(list(interpolateAB(model, trees[i1], trees[i2], 5)))
            selected_pairs.append((i1, i2))
            i += 1
    '''for i in range(len(trees)):
        for j in range(i+1, len(trees)):
            print(i, j)
            if zss.simple_distance(trees[i], trees[j], get_label=Node.get_symbol, label_dist=symbol_distance) < 10: continue
            results.append(list(interpolateAB(model, trees[i], trees[j], 5)))'''
    results = np.array(results)
    print(np.mean(results, axis=0))
    with open('../data/linear_interpolation_results_nhvae.pkl', 'wb') as f:
        pickle.dump(results, f)
    f.close()
    input("done")

    # Expressions we want to interpolate between
    #exprA = "log ( X_0 ) / ( X_0 + X_0 )"
    #exprB = "sqrt ( C * sin ( X_0 ) ) * X_0 * X_0"

    exprA = "log ( C * C / X_0 ) - X_0 - X_0 ^2"
    exprB = "C + X_0 * C - X_0 * X_0"

    #exprA = "A * A / C * cos ( C * A )"
    #exprB = "A - ( C + A + C ) / C - A"

    # Number of steps in the interpolation (inclusive with expressions A and B)
    steps = 5

    tokensA = exprA.split(" ")
    tokensB = exprB.split(" ")
    treeA = tokens_to_tree(tokensA, so, data_config["max_arity"])
    treeB = tokens_to_tree(tokensB, so, data_config["max_arity"])

    interpolateAB(model, treeA, treeB, steps)
