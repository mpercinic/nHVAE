from argparse import ArgumentParser
import json

from ProGED.generators.grammar import GeneratorGrammar

from tree import Node
from utils import load_config_file
from symbol_library import generate_symbol_library, SymType


def generate_expr_grammar(symbols):
    grammar = ""
    variables = []
    functions = []
    constants = False
    for symbol in symbols:
        if symbol["type"].value == SymType.Var.value:
            variables.append(symbol["symbol"])
        elif symbol["type"].value == SymType.Fun.value and symbol['key'][0] != '^':
            functions.append(symbol["symbol"])
        elif symbol["type"].value == SymType.Const.value:
            constants = True
    grammar += "E -> E '+' F [0.2]\n"
    grammar += "E -> E '-' F [0.2]\n"
    grammar += "E -> F [0.6]\n"

    grammar += "F -> F '*' T [0.2]\n"
    grammar += "F -> F '/' T [0.2]\n"
    grammar += "F -> T [0.6]\n"

    grammar += "T -> A [0.3]\n"
    remaining = 0.7
    if constants:
        grammar += "T -> 'C' [0.3]\n"
        remaining -= 0.3
    grammar += f"T -> V [{remaining}]\n"

    var_prob = 1 / len(variables)
    for v in variables:
        grammar += f"V -> '{v}' [{var_prob}]\n"

    fun_prob = 1 / len(functions)
    remaining = 1
    if len(functions) > 0:
        grammar += "A -> R '(' E ')' [0.35]\n"
        remaining -= 0.35
    grammar += "A -> '(' E ')' P [0.1]\n"
    remaining -= 0.1
    grammar += f"A -> '(' E ')' [{remaining}]\n"

    if len(functions) > 0:
        for f in functions:
            grammar += f"R -> '{f}' [{fun_prob}]\n"

    grammar += "P -> '^2' [0.8]\n"
    grammar += "P -> '^3' [0.2]\n"

    return grammar

def generate_neuro_grammar():
    grammar = """Node -> Pop1 CF SCF CM1 [0.5]
    Node -> Pops2 CF SCF SCF CM2 [0.28]
    Node -> Pop1 Pops2 CF SCF SCF SCF CM [0.06]
    Node -> Pops2 Pops2 CF SCF SCF SCF SCF CM [0.06]
    Node -> Pops2 Pops2 Pop1 CF SCF SCF SCF SCF SCF CM [0.1]
    Pops2 -> Pop1 Pop1 [0.78]
    Pops2 -> Pop2 [0.22]
    Pop1 -> InputDyn OutputDyn ECF ECF SN [1.0]
    Pop2 -> InputDyn InputDyn OutputDyn ECF ECF ECF ECF SN [1.0]
    InputDyn -> S S [0.14]
    InputDyn -> 'second_order_kernel' [0.66]
    InputDyn -> 'exp_kernel' [0.08]
    InputDyn -> 'gating_kinetics' [0.06]
    InputDyn -> 'linear_kernel' [0.03]
    InputDyn -> 'voltage_gated_dynamics' [0.03]
    OutputDyn -> 'direct_readout' [0.84]
    OutputDyn -> 'membrane_integrator' [0.06]
    OutputDyn -> 'difference' [0.06]
    OutputDyn -> 'spatial_gradient' [0.04]
    ECF -> 'linear' [0.53]
    ECF -> 'custom' [0.15]
    ECF -> 'false' [0.32]
    SCF -> 'custom' [0.89]
    SCF -> 'linear' [0.11]
    CF -> 'saturating_sigmoid' [0.52]
    CF -> 'relaxed_rectifier' [0.24]
    CF -> 'baseline_sigmoid' [0.24]
    S -> P '+' S [0.5]
    S -> P [0.5]
    P -> V '*' P [0.42]
    P -> V [0.58]
    V -> 'x1' [0.56]
    V -> 'x2' [0.44]
    CM1 -> 'full' [0.2]
    CM1 -> 'null' [0.8]
    CM2 -> 'full' [0.5]
    CM2 -> 'ring' [0.3]
    CM2 -> Digit [0.2]
    CM -> 'full' [0.06]
    CM -> 'ring' [0.06]
    CM -> 'star' [0.4]
    CM -> 'hub_tail' [0.06]
    CM -> 'star_feedback_tail' [0.06]
    CM -> 'star_loop_extended' [0.15]
    CM -> 'ei_extended' [0.15]
    CM -> 'small_world' Digit Digit [0.06]
    Digit -> '0' [0.1]
    Digit -> '1' [0.1]
    Digit -> '2' [0.1]
    Digit -> '3' [0.1]
    Digit -> '4' [0.1]
    Digit -> '5' [0.1]
    Digit -> '6' [0.1]
    Digit -> '7' [0.1]
    Digit -> '8' [0.1]
    Digit -> '9' [0.1]
    SN -> 'true' [0.17]
    SN -> 'false' [0.83]"""

    return grammar


def is_float(element: any) -> bool:
    #If you expect None to be passed:
    if element is None:
        return False
    elif element[0] == '+':
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False


def tokens_to_tree(tokens, symbols, max_arity):
    start_expr = "".join(tokens)
    num_tokens = len([t for t in tokens if t != "(" and t != ")"])
    tokens = ["("] + tokens + [")"]
    operator_stack = []
    out_stack = []
    children = []
    for token in tokens:
        if token == "(":
            operator_stack.append(token)
        elif token in symbols and (symbols[token]["type"].value == SymType.Var.value or symbols[token]["type"].value == SymType.Const.value) or is_float(token):
            out_stack.append(Node(token, children=[]))
        elif token in symbols and symbols[token]["type"].value == SymType.Fun.value:
            if token[0] == "^":
                out_stack.append(Node(token, children=[out_stack.pop()]))
            else:
                operator_stack.append(token)
        elif token in symbols and symbols[token]["type"].value == SymType.Operator.value:
            while len(operator_stack) > 0 and operator_stack[-1] != '(' \
                    and (symbols[operator_stack[-1]]["precedence"] > symbols[token]["precedence"]
                        or (symbols[operator_stack[-1]]["precedence"] == symbols[token]["precedence"]
                            and "arity" in symbols[operator_stack[-1]])):
                if symbols[operator_stack[-1]]["type"].value == SymType.Fun.value:
                    out_stack.append(Node(operator_stack.pop(), children=[out_stack.pop()]))
                else:
                    op_current = operator_stack[-1]
                    if operator_stack[-1] == '*' and len(operator_stack) > 1 and operator_stack[-2] == op_current \
                            and len(children) < max_arity - 2:
                        children.append(out_stack.pop())
                        operator_stack.pop()
                    else:
                        children.append(out_stack.pop())
                        children.append(out_stack.pop())
                        children.reverse()
                        symbol = operator_stack.pop()
                        out_stack.append(Node(symbol, children=children))
                        children = []
            operator_stack.append(token)
        else:
            while len(operator_stack) > 0 and operator_stack[-1] != '(':
                if symbols[operator_stack[-1]]["type"].value == SymType.Fun.value:
                    out_stack.append(Node(operator_stack.pop(), children=[out_stack.pop()]))
                else:
                    op_current = operator_stack[-1]
                    if (operator_stack[-1] == '+' or operator_stack[-1] == '*') and len(operator_stack) > 1 \
                            and operator_stack[-2] == op_current and len(children) < max_arity - 2:
                        children.append(out_stack.pop())
                        operator_stack.pop()
                    else:
                        children.append(out_stack.pop())
                        children.append(out_stack.pop())
                        children.reverse()
                        symbol = operator_stack.pop()
                        out_stack.append(Node(symbol, children=children))
                        children = []
            operator_stack.pop()
            if len(operator_stack) > 0 and operator_stack[-1] in symbols and symbols[operator_stack[-1]]["type"].value == SymType.Fun.value:
                out_stack.append(Node(operator_stack.pop(), children=[out_stack.pop()]))
    if len(out_stack[-1].to_list(dataset)) < num_tokens:
        raise Exception(f"Could not parse the whole expression {start_expr}")
    return out_stack[-1]


def generate_expressions(grammar, number_of_all_expressions, symbols, max_arity, max_length):
    generator = GeneratorGrammar(grammar)
    expression_set = set()
    expression_trees = []

    while len(expression_trees) < number_of_all_expressions:
        if len(expression_trees) % 500 == 0:
            print(f"Unique expressions generated so far: {len(expression_trees)}")
        expr = generator.generate_one()[0]

        try:
            expr_tree = tokens_to_tree(expr, symbols, max_arity)
            expr_str = "".join(expr_tree.to_list(dataset))
            if expr_str in expression_set:
                continue
            if len([s for s in expr_tree.to_list(dataset) if s not in ["(", ")"]]) > max_length:
                continue
        except:
            continue
        expression_trees.append(expr_tree)
        expression_set.add(expr_str)
    return expression_trees

def prods_to_tree(prods):
    symbol = str(prods[0]).split(" ")[0]
    children_symbols = [cs.replace("\'", "") for cs in str(prods[0]).split(" ")[2:-1]]
    children_nonterminal = []
    for child in prods[1:]:
        children_nonterminal.append(str(child[0]).split(" ")[0])
    children = []
    terminalcounter = 0
    for i in range(len(children_symbols)):
        if children_symbols[i] not in children_nonterminal:
            children.append(Node(children_symbols[i], []))
            terminalcounter += 1
        else:
            children.append(prods_to_tree(prods[i-terminalcounter+1]))
    return Node(symbol, children)

def generate_parse_trees(grammar, number_of_all_expressions, max_length):
    generator = GeneratorGrammar(grammar)
    parsetree_set = set()
    parse_trees = []

    while len(parse_trees) < number_of_all_expressions:
        if len(parse_trees) % 500 == 0:
            print(f"Unique trees generated so far: {len(parse_trees)}")
        ptree = generator.generate_one()[3]
        if ptree in parsetree_set: continue

        tree = prods_to_tree(ptree)
        if len(tree) > max_length: continue

        parse_trees.append(tree)
        parsetree_set.add(ptree)

    return parse_trees

def symbol_distance(a, b):
    return int(a != b)


if __name__ == '__main__':
    parser = ArgumentParser(prog='Data set generation', description='Generate a set of trees')
    parser.add_argument("-config", default="../configs/config.json")
    args = parser.parse_args()

    config = load_config_file(args.config)
    data_config = config["data_definition"]
    ds_config = config["data_set_generation"]
    dataset = config["dataset"]

    symbols = data_config["expr_symbols"] if dataset == "expr" else data_config["neuro_symbols"]
    sy_lib = generate_symbol_library(data_config["num_variables"], symbols, dataset, data_config["has_constants"])
    Node.add_symbols(sy_lib)
    so = {s["symbol"]: s for s in sy_lib} if dataset == "expr" else {s["key"]: s for s in sy_lib}

    # Optional (recommended): Generate training set from a custom grammar
    grammar = None

    if grammar is None:
        grammar = generate_expr_grammar(sy_lib) if dataset == "expr" else generate_neuro_grammar()

    trees = generate_expressions(grammar, ds_config["num_trees"], so, data_config["max_arity"], ds_config["max_length"]) \
        if dataset == "expr" else generate_parse_trees(grammar, ds_config["num_trees"], ds_config["max_length"])
    print("Number of expressions generated: " + str(len(trees)))

    expr_dict = [tree.to_dict() for tree in trees]

    save_path = ds_config["data_set_path"]
    if save_path != "":
        with open(save_path, "w") as file:
            json.dump(expr_dict, file)
