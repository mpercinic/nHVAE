from argparse import ArgumentParser

import torch

from model import nHVAE
from utils import tokens_to_tree, load_config_file, create_batch
from symbol_library import generate_symbol_library


def interpolateAB(model, treeA, treeB, steps):
    treeBA = create_batch([treeA])
    treeBB = create_batch([treeB])
    l1 = model.encode(treeBA)[0]
    l2 = model.encode(treeBB)[0]
    print(f"Expr A:\t{treeA.to_string('expr')}")
    print()
    print(f"a=0:\t{model.decode(l1)[0].to_string('expr')}")
    for i in range(1, steps-1):
        a = i/(steps-1)
        la = (1-a) * l1 + a * l2
        print(f"a={str(a)[:5]}:\t{model.decode(la)[0].to_string('expr')}")
    print(f"a=1:\t{model.decode(l2)[0].to_string('expr')}")
    print()
    print(f"Expr B:\t{treeB.to_string('expr')}")


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

    # Expressions we want to interpolate between
    exprA = "log ( A ) / ( A + A )"
    exprB = "sqrt ( C * sin ( A ) ) * A * A"

    # Number of steps in the interpolation (inclusive with expressions A and B)
    steps = 5

    tokensA = exprA.split(" ")
    tokensB = exprB.split(" ")
    treeA = tokens_to_tree(tokensA, so, data_config["max_arity"])
    treeB = tokens_to_tree(tokensB, so, data_config["max_arity"])

    interpolateAB(model, treeA, treeB, steps)
