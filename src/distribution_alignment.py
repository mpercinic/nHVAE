from argparse import ArgumentParser
import numpy as np
import torch
import json

from utils import read_trees_json, load_config_file
from symbol_library import generate_symbol_library
from tree import Node
from expression_set_generation import tokens_to_tree

def decode_generated_set(model, dataset_size, path):
    dataset = [torch.normal(mean=torch.zeros(128), std=torch.ones(128)) for _ in range(dataset_size)]
    tree_dataset = []
    for i in range(int(dataset_size / 32)):
        z = torch.stack(dataset[i*32:(i+1)*32])[:, None, :]
        decoded_trees = model.decode(z)
        for tree in decoded_trees:
            tree_dataset.append(tree)

    expr_dict = [tree.to_dict() for tree in tree_dataset]

    with open(path.split(".json")[0] + '_decoded.json', "w") as file:
        json.dump(expr_dict, file)
    file.close()


if __name__ == '__main__':
    parser = ArgumentParser(prog='Structural distribution alignment', description='Compute structural distribution alignment')
    parser.add_argument("-config", default="../configs/config.json")
    args = parser.parse_args()

    config = load_config_file(args.config)
    data_config = config["data_definition"]
    ds_config = config["data_set_generation"]
    training_config = config["training"]
    datasetstr = config["dataset"]
    path = ds_config["data_set_path"]

    if training_config["seed"] is not None:
        np.random.seed(training_config["seed"])
        torch.manual_seed(training_config["seed"])

    model = torch.load(training_config["param_path"])
    decode_generated_set(model, ds_config["num_trees"], path)

    trees = read_trees_json(path)
    trees_decoded = read_trees_json(path.split(".json")[0] + '_decoded.json')
    if datasetstr == "expr":
        symbols = data_config["expr_symbols"] if datasetstr == "expr" else data_config["neuro_symbols"]
        sy_lib = generate_symbol_library(data_config["num_variables"], symbols, datasetstr, data_config["has_constants"])
        Node.add_symbols(sy_lib)
        so = {s["symbol"]: s for s in sy_lib} if datasetstr == "expr" else {s["key"]: s for s in sy_lib}
        trees_decoded = [tokens_to_tree(t.to_list(), so, 5) for t in trees_decoded]
    lens = [len(t) for t in trees]
    lens_decoded = [len(t) for t in trees_decoded]
    maxlen = max(lens + lens_decoded)

    heights = [t.height() for t in trees]
    heights_decoded = [t.height() for t in trees_decoded]
    maxheight = max(heights + heights_decoded)

    avg_branchings = [round((len(t) - 1) / t.n_nonleaf_nodes(), 1) if len(t) > 1 else 0 for t in trees]
    avg_branchings_decoded = [round((len(t) - 1) / t.n_nonleaf_nodes(), 1) if len(t) > 1 else 0 for t in trees_decoded]
    max_bf = max(avg_branchings + avg_branchings_decoded)

    lengths_dist, lengths_decoded_dist = [0] * maxlen, [0] * maxlen
    heights_dist, heights_decoded_dist = [0] * maxheight, [0] * maxheight
    branchings_dist, branchings_decoded_dist = [0] * (round(max_bf / 0.1) + 1), [0] * (round(max_bf / 0.1) + 1)

    for i in range(len(trees)):
        lengths_dist[lens[i] - 1] += 1
        heights_dist[heights[i] - 1] += 1
        branchings_dist[round(avg_branchings[i] / 0.1)] += 1

    for i in range(len(trees_decoded)):
        lengths_decoded_dist[lens_decoded[i] - 1] += 1
        heights_decoded_dist[heights_decoded[i] - 1] += 1
        branchings_decoded_dist[round(avg_branchings_decoded[i] / 0.1)] += 1

    print(lengths_dist)
    print(lengths_decoded_dist)
    print((40000 - sum(lengths_decoded_dist)) / 40000)

    lengths_dist = [i / 40000 for i in lengths_dist]
    heights_dist = [i / 40000 for i in heights_dist]
    branchings_dist = [i / 40000 for i in branchings_dist]
    lengths_decoded_dist = [i / sum(lengths_decoded_dist) for i in lengths_decoded_dist]
    heights_decoded_dist = [i / sum(heights_decoded_dist) for i in heights_decoded_dist]
    branchings_decoded_dist = [i / sum(branchings_decoded_dist) for i in branchings_decoded_dist]

    print('WD length: ' + str(wasserstein_distance(lengths_dist, lengths_decoded_dist)))
    print('WD height: ' + str(wasserstein_distance(heights_dist, heights_decoded_dist)))
    print('WD bf: ' + str(wasserstein_distance(branchings_dist, branchings_decoded_dist)))


