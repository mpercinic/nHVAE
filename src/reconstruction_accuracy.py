import random
from argparse import ArgumentParser

import numpy as np
import torch
from sklearn.model_selection import KFold
import zss

from utils import read_trees_json, load_config_file, create_batch
from symbol_library import generate_symbol_library
from model import nHVAE
from train import train_hvae
from tree import Node
from expression_set_generation import tokens_to_tree, generate_expr_grammar, generate_neuro_grammar
from validity_checking import *

def symbol_distance(s1, s2):
    return int(s1 != s2)

def one_fold(model, train, test, epochs, batch_size, verbose, symbols, max_arity, grammar, invalid):
    train_hvae(model, train, epochs, batch_size, verbose)

    total_distance = []
    counter = 0
    for i in range((len(test) // batch_size) + 1):
        batch = create_batch(test[(i*batch_size):((i+1)*batch_size)])
        latent = model.encode(batch)[0]
        pts = model.decode(latent)
        for j in range(len(pts)):
            if dataset == 'expr':
                expr_tree = tokens_to_tree(test[i * batch_size + j].to_list(dataset), symbols, max_arity)
                decoded_tree = tokens_to_tree(pts[j].to_list(dataset), symbols, max_arity)
                if pts[j].to_pexpr() != decoded_tree.to_pexpr() and test[i * batch_size + j].to_pexpr() == expr_tree.to_pexpr():
                    counter += 1
                total_distance.append(zss.simple_distance(expr_tree, decoded_tree, get_label=Node.get_symbol, label_dist=symbol_distance))
            elif dataset == 'neuro':
                if not pts[j].verify(grammar):
                    invalid += 1
                    continue
                total_distance.append(zss.simple_distance(test[i * batch_size + j], pts[j], get_label=Node.get_symbol, label_dist=symbol_distance))
    print(len(total_distance))
    print(counter)
    return total_distance, invalid

def one_experiment(name, trees, input_dim, latent_dim, epochs, batch_size, verbose, seed, max_arity, symbols, grammar,
                   smaller_dataset=False, examples=2000, n_splits=5, results_path=None):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    distances = []
    n_invalid = 0
    for i, (train_idx, test_idx) in enumerate(kf.split(trees)):
        train_idx = np.random.permutation(train_idx)
        print(f"Fold {i + 1}")
        if smaller_dataset:
            np.random.seed(seed + i)
            torch.manual_seed(seed + i)
            inds = train_idx[:examples]
            train = [trees[i] for i in inds]
        else:
            train = [trees[i] for i in train_idx]

        test = [trees[i] for i in test_idx]
        model = nHVAE(input_dim, latent_dim, max_arity, dataset=dataset)
        d, inv = one_fold(model, train, test, epochs, batch_size, verbose, symbols, max_arity, grammar, 0)
        distances.append(d)
        n_invalid += inv
        print(f"Mean: {np.mean(distances[-1])}, Var: {np.var(distances[-1])}")
        print("Invalid: " + str(inv) + '/' + str(len(test_idx)))
        print()
    fm = [np.mean(d) for d in distances]
    print(f"Mean: {np.mean(fm)}, Std dev: {np.std(fm)}, All: {', '.join([str(f) for f in fm])}")
    print("Invalid: " + str(n_invalid))
    if results_path is not None:
        with open(results_path, "a") as file:
            file.write(f"{name}\t Mean: {np.mean(fm)}, Std dev: {np.std(fm)}, All: {', '.join([str(f) for f in fm])}\n")
    return fm

if __name__ == '__main__':
    parser = ArgumentParser(prog='Tree reconstruction', description='Evaluate the reconstruction ability of nHVAE')
    parser.add_argument("-config", default="../configs/config.json")
    args = parser.parse_args()

    config = load_config_file(args.config)
    data_config = config["data_definition"]
    ds_config = config["data_set_generation"]
    training_config = config["training"]
    reconstruction_config = config["reconstruction"]
    dataset = config["dataset"]

    if training_config["seed"] is not None:
        np.random.seed(training_config["seed"])
        torch.manual_seed(training_config["seed"])
        random.seed(training_config["seed"])

    symbols = data_config["expr_symbols"] if dataset == "expr" else data_config["neuro_symbols"]
    sy_lib = generate_symbol_library(data_config["num_variables"], symbols, dataset, data_config["has_constants"])

    nHVAE.add_symbols(sy_lib)
    Node.add_symbols(sy_lib)
    so = {s["symbol"]: s for s in sy_lib} if dataset == "expr" else {s["key"]: s for s in sy_lib}

    trees = read_trees_json(ds_config["data_set_path"])
    max_arity = max([t.max_branching_factor() for t in trees])
    random.shuffle(trees)

    grammar = generate_expr_grammar(sy_lib) if dataset == "expr" else generate_neuro_grammar()
    if dataset == "neuro":
        constraints = {
            "Node_post": lambda grammar, rule, constraints: constraint_Node(grammar, rule, constraints),
            "NumIncomingConns_pre": lambda grammar, constraints: constraint_NumIncomingConns(grammar, constraints),
            "ConnFuncMode_pre": lambda grammar, constraints: constraint_ConnFuncMode(grammar, constraints)
        }
        grammar = Grammar(grammar, "Node", constraints)

    one_experiment(ds_config["data_set_path"], trees, len(sy_lib), training_config["latent_size"],
                   training_config["epochs"], training_config["batch_size"], training_config["verbose"],
                   training_config["seed"], max_arity, so, grammar, reconstruction_config["smaller_dataset"],
                   reconstruction_config["num_trees"], reconstruction_config["n_folds"],
                   reconstruction_config["results_path"])

