from argparse import ArgumentParser

import numpy as np
import torch
from torch.utils.data import Sampler, Dataset
from tqdm import tqdm
import pickle

from hvae_utils import read_expressions_json, load_config_file, create_batch
from model import HVAE
from symbol_library import generate_symbol_library


def collate_fn(batch):
    return batch


class TreeBatchSampler(Sampler):
    def __init__(self, batch_size, num_eq):
        self.batch_size = batch_size
        self.num_eq = num_eq
        self.permute = np.random.permutation(self.num_eq)

    def __iter__(self):
        for i in range(len(self)):
            batch = self.permute[(i*self.batch_size):((i+1)*self.batch_size)]
            yield batch

    def __len__(self):
        return self.num_eq // self.batch_size


class TreeDataset(Dataset):
    def __init__(self, train):
        self.train = train

    def __getitem__(self, idx):
        return self.train[idx]

    def __len__(self):
        return len(self.train)


def logistic_function(iter, total_iters, supremum=0.045):
    x = iter/total_iters
    return supremum/(1+50*np.exp(-10*x))


def train_hvae(model, trees, epochs=20, batch_size=32, verbose=True):
    dataset = TreeDataset(trees)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")

    iter_counter = 0
    total_iters = epochs*(len(dataset)//batch_size)
    lmbda = logistic_function(iter_counter, total_iters)
    # lmbda = (np.tanh(-4.5) + 1) / 2

    midpoint = len(dataset) // (2 * batch_size)

    for epoch in range(epochs):
        sampler = TreeBatchSampler(batch_size, len(dataset))
        bce, kl, total, num_iters = 0, 0, 0, 0

        with tqdm(total=len(dataset), desc=f'Testing - Epoch: {epoch + 1}/{epochs}', unit='chunks') as prog_bar:
            for i, tree_ids in enumerate(sampler):
                batch = create_batch([dataset[j] for j in tree_ids])

                mu, logvar, outputs = model(batch)
                loss, bcel, kll = outputs.loss(mu, logvar, lmbda, criterion)
                bce += bcel.detach().item()
                kl += kll.detach().item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                num_iters += 1
                prog_bar.set_postfix(**{'run:': "HVAE",
                                        'loss': (bce+kl) / num_iters,
                                        'BCE': bce / num_iters,
                                        'KLD': kl / num_iters})
                prog_bar.update(batch_size)

                lmbda = logistic_function(iter_counter, total_iters)
                iter_counter += 1
                '''if iter_counter < 9000:
                    lmbda = (np.tanh((0.3333 * iter_counter - 4500) / 1000) + 1) / 2'''

                if verbose and i == midpoint:
                    original_trees = batch.to_expr_list()
                    z = model.encode(batch)[0]
                    filename = "test.pkl"
                    with open(filename, "wb") as f:
                        pickle.dump(z, f)
                    decoded_trees = model.decode(z)
                    for i in range(len(decoded_trees)):
                        print()
                        # print(original_trees[i].to_pexpr())
                        print(f"O: {original_trees[i]}")
                        print(f"P: {decoded_trees[i]}")
                        # print(decoded_trees[i].to_pexpr())


if __name__ == '__main__':
    parser = ArgumentParser(prog='Model training', description='Train a HVAE model')
    parser.add_argument("-config", default="../configs/test_config.json")
    args = parser.parse_args()

    config = load_config_file(args.config)
    expr_config = config["expression_definition"]
    es_config = config["expression_set_generation"]
    training_config = config["training"]

    if training_config["seed"] is not None:
        np.random.seed(training_config["seed"])
        torch.manual_seed(training_config["seed"])

    extra_symbols = []
    for i in range(2, expr_config["max_arity"]):
        extra_symbols += ["+" + str(i), "*" + str(i)]

    sy_lib, _ = generate_symbol_library(expr_config["num_variables"], expr_config["symbols"] + extra_symbols,
                                        expr_config["max_arity"], expr_config["has_constants"])

    HVAE.add_symbols(sy_lib)

    trees = read_expressions_json(es_config["expression_set_path"])

    model = HVAE(len(sy_lib), training_config["latent_size"], expr_config["max_arity"] - 1)

    '''with open("test.pkl", 'rb') as f:
        z = pickle.load(f)
    decoded_trees = model.decode(z)
    for i in range(len(decoded_trees)):
        if decoded_trees[i] is not None:
            print()
            #print(original_trees[i].to_pexpr())
            #print(f"O: {original_trees[i]}")
            print(f"P: {decoded_trees[i]}")'''

    train_hvae(model, trees, training_config["epochs"], training_config["batch_size"], training_config["verbose"])

    if training_config["param_path"] != "":
        torch.save(model, training_config["param_path"])
