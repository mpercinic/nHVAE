from argparse import ArgumentParser

import torch

from model import HVAE
from hvae_utils import tokens_to_tree, load_config_file, create_batch
from symbol_library import generate_symbol_library


def interpolateAB(model, treeA, treeB, steps=5):
    treeBA = create_batch([treeA])
    treeBB = create_batch([treeB])
    l1 = model.encode(treeBA)[0]
    l2 = model.encode(treeBB)[0]
    print(f"Expr A:\t{str(treeA)}")
    print(f"a=0:\t{str(model.decode(l1)[0])}")
    for i in range(1, steps-1):
        a = i/(steps-1)
        la = (1-a) * l1 + a * l2
        print(f"a={str(a)[:5]}:\t{str(model.decode(la)[0])}")
    print(f"a=1:\t{str(model.decode(l2)[0])}")
    print(f"Expr B:\t{str(treeB)}")


if __name__ == '__main__':
    parser = ArgumentParser(prog='Linear interpolation', description='Interpolate between two expressions')
    parser.add_argument("-config", default="../configs/test_config.json")
    args = parser.parse_args()

    config = load_config_file(args.config)
    expr_config = config["expression_definition"]
    es_config = config["expression_set_generation"]
    training_config = config["training"]

    extra_symbols = []
    for i in range(2, expr_config["max_arity"]):
        extra_symbols += ["+" + str(i), "*" + str(i)]
    sy_lib, sy_lib_basic = generate_symbol_library(expr_config["num_variables"], expr_config["symbols"] + extra_symbols,
                                        expr_config["max_arity"], expr_config["has_constants"])
    so = {s["symbol"]: s for s in sy_lib_basic}
    HVAE.add_symbols(sy_lib)

    model = torch.load(training_config["param_path"])

    # Expressions we want to interpolate between
    exprA = "cos ( A * A + A ) + exp ( A ) / A"
    exprB = "A ^2 - A ^3"
    # Number of steps in the interpolation (inclusive with expressions A and B)
    steps = 5

    tokensA = exprA.split(" ")
    tokensB = exprB.split(" ")
    treeA = tokens_to_tree(tokensA, so)
    treeB = tokens_to_tree(tokensB, so)

    interpolateAB(model, treeA, treeB)
