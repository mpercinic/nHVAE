from argparse import ArgumentParser
import json

import numpy as np
from ProGED.generators.grammar import GeneratorGrammar

from tree import Node
from hvae_utils import load_config_file, read_expressions_json
from symbol_library import generate_symbol_library, SymType

import pickle
import editdistance
import zss


def generate_grammar(symbols):
    grammar = ""
    operators = {}
    functions = []
    powers = []
    variables = []
    constants = False
    for symbol in symbols:
        if symbol["type"].value == SymType.Operator.value:
            if symbol["precedence"] in operators:
                operators[symbol["precedence"]].append(symbol["symbol"])
            else:
                operators[symbol["precedence"]] = [symbol["symbol"]]
        elif symbol["type"].value == SymType.Fun.value and symbol["precedence"] < 0:
            powers.append(symbol["symbol"])
        elif symbol["type"].value == SymType.Fun.value:
            functions.append(symbol["symbol"])
        elif symbol["type"].value == SymType.Var.value:
            variables.append(symbol["symbol"])
        elif symbol["type"].value == SymType.Const.value:
            constants = True
        else:
            raise Exception("Error during generation of the grammar")

    if 0 in operators:
        '''op_prob = 0.4 / len(operators[0])
        for op in operators[0]:
            grammar += f"E -> E '{op}' F [{op_prob}]\n"'''
        # new grammar
        grammar += "E -> E '-' F [0.1]\n"
        grammar += "E -> E '+' F [0.04]\n"
        grammar += "E -> E '+' F '+' F [0.03]\n"
        grammar += "E -> E '+' F '+' F '+' F [0.02]\n"
        grammar += "E -> E '+' F '+' F '+' F '+' F [0.01]\n"
        grammar += "E -> F [0.8]\n"
        # old grammar
        '''grammar += "E -> E '+' F [0.2]\n"
        grammar += "E -> E '-' F [0.2]\n"
        grammar += "E -> F [0.6]\n"'''
    else:
        grammar += "E -> F [1.0]\n"

    if 1 in operators:
        '''op_prob = 0.4 / len(operators[1])
        for op in operators[1]:
            grammar += f"F -> F '{op}' T [{op_prob}]\n"'''
        # new grammar
        grammar += "F -> F '/' T [0.1]\n"
        grammar += "F -> F '*' T [0.04]\n"
        grammar += "F -> F '*' T '*' T [0.03]\n"
        grammar += "F -> F '*' T '*' T '*' T [0.02]\n"
        grammar += "F -> F '*' T '*' T '*' T '*' T [0.01]\n"
        grammar += "F -> T [0.8]\n"
        # old grammar
        '''grammar += "F -> F '*' T [0.2]\n"
        grammar += "F -> F '/' T [0.2]\n"
        grammar += "F -> T [0.6]\n"'''
    else:
        grammar += "F -> T [1.0]\n"

    remaining = 1
    if len(powers) > 0:
        grammar += "T -> '(' E ')' P [0.2]\n"
        remaining -= 0.2

    if len(functions) > 0:
        grammar += "T -> R '(' E ')' [0.2]\n"
        remaining -= 0.2

    # for reconstruction accuracy
    grammar += f"T -> '(' E ')' '^' '(' E ')' [0.1]\n"
    remaining -= 0.1

    remaining /= 4
    grammar += f"T -> V [{3 * remaining}]\n"
    grammar += f"T -> '(' E ')' [{remaining}]\n"

    var_prob = 1 / len(variables) if not constants else 1 / (len(variables) + 1)
    for v in variables:
        grammar += f"V -> '{v}' [{var_prob}]\n"

    if constants:
        grammar += f"V -> 'C' [{var_prob}]\n"

    if len(functions) > 0:
        function_prob = 1 / len(functions)
        for funct in functions:
            grammar += f"R -> '{funct}' [{function_prob}]\n"

    if len(powers) > 0:
        powers = sorted(powers)
        power_probs = [1 / (1 + int(p[1:])) for p in powers]
        power_probs = np.array(power_probs) / sum(power_probs)
        for p, prob in zip(powers, power_probs):
            grammar += f"P -> '{p}' [{prob}]\n"

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
    """
    tokens : list of string tokens
    symbols: dictionary of possible tokens -> attributes, each token must have attributes: nargs (0-2), order
    """
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
                        if symbol == '+' or symbol == '*':
                            symbol += str(len(children))
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
                        if symbol == '+' or symbol == '*':
                            symbol += str(len(children))
                        out_stack.append(Node(symbol, children=children))
                        children = []
            operator_stack.pop()
            if len(operator_stack) > 0 and operator_stack[-1] in symbols and symbols[operator_stack[-1]]["type"].value == SymType.Fun.value:
                out_stack.append(Node(operator_stack.pop(), children=[out_stack.pop()]))
    if len(out_stack[-1].to_list()) < num_tokens:
        raise Exception(f"Could not parse the whole expression {start_expr}")
    return out_stack[-1]


def generate_expressions(grammar, number_of_all_expressions, symbols, max_arity, max_depth):
    generator = GeneratorGrammar(grammar)
    expression_set = set()
    #expressions = []
    expressions_final = []
    #for j in range(max_depth):
    #    expressions.append([])
    #with open("../expressions.pkl", 'rb') as f:
    #    exprs = pickle.load(f)
    #with open("../expressions_c_equal.pkl", 'rb') as f:
    #    exprs = pickle.load(f)
    '''exprs = open("../data/expressions_5_15k.txt")
    for expr in exprs.readlines():
        expr = expr[:-1].split(" ")
        if len(expression_set) % 500 == 0:
            print(f"Unique expressions generated so far: {len(expression_set)}")
        #if has_constants:
        #    pass

        expr_str = "".join(expr)

        try:
            expr_tree = tokens_to_tree(expr, symbols, max_arity)
            expressions_final.append(expr_tree)
            expression_set.add(expr_str)
        except:
            continue
    exprs.close()'''

    '''trees = read_expressions_json("../data/expression_sets/ng_newgrammar.json")
    for t in trees:
        expr = t.to_list()

        expr_tree = tokens_to_tree(expr, symbols, max_arity)
        expressions_final.append(expr_tree)'''

    '''trees = read_expressions_json("../data/expression_sets/ptrees_5_15k_grammar_trig.json")

    lengths = [0] * 32
    lengths_new = [0] * 32
    for tree in trees:
        lengths[len(tree.to_list()) - 1] += 1
    print(lengths)
    while len(expression_set) < number_of_all_expressions:
        if len(expression_set) % 500 == 0:
            print(f"Unique expressions generated so far: {len(expression_set)}")
        expr = generator.generate_one()[0]

        try:
            expr_tree = tokens_to_tree(expr, symbols, max_arity)
            expr_str = "".join(expr_tree.to_list())
            length = len(expr_tree.to_list())
            if expr_str in expression_set:
                continue
            if expr_tree.height() > max_depth or length > 32:  # or expr_length > max_length:
                continue
            #if lengths_new[length-1] == lengths[length-1]:
            #    continue
            expressions_final.append(expr_tree)
            expression_set.add(expr_str)
            lengths_new[length-1] += 1
        except:
            continue'''

    while len(expression_set) < number_of_all_expressions:
        if len(expression_set) % 500 == 0:
            print(f"Unique expressions generated so far: {len(expression_set)}")
            if len(expressions_final) > 0:
                print("".join(expressions_final[-1].to_list()))
        expr = generator.generate_one()[0]

        try:
            expr_tree = tokens_to_tree(expr, symbols, max_arity)
            #expr_length = 0
            #for i in expr:
            #    if i != '(' and i != ')':
            #        expr_length += 1
            #tree_height = expr_tree.height()
            expr_str = "".join(expr_tree.to_list())
            if expr_str in expression_set:
                continue
            #max_bf = expr_tree.max_branching_factor()
            if len([s for s in expr_tree.to_list() if s not in ["(", ")"]]) > 42:  # tree_height > max_depth or
                continue

            #expressions[max_bf].append(expr_tree)
            expressions_final.append(expr_tree)
            expression_set.add(expr_str)
        except:
            continue
    #with open("all_expressions.pkl", "wb") as f:
    #    pickle.dump(expressions, f)
    '''expressions_final = []
    for j in range(len(expressions)):
        length = len(expressions[j])
        print(length)
        stop = round(length*number_of_expressions/number_of_all_expressions)
        sample = np.random.permutation(length)
        for k in range(stop):
            expressions_final.append(expressions[j][sample[k]])'''
    #print(lengths_new)
    #input()
    return expressions_final

def symbol_distance(a, b):
    return int(a != b)


if __name__ == '__main__':
    parser = ArgumentParser(prog='Expression set generation', description='Generate a set of expressions')
    parser.add_argument("-config", default="../configs/test_config.json")
    args = parser.parse_args()

    config = load_config_file(args.config)
    expr_config = config["expression_definition"]
    es_config = config["expression_set_generation"]

    extra_symbols = []
    for i in range(2, expr_config["max_arity"] + 1):
        extra_symbols += ["+" + str(i), "*" + str(i)]
    sy_lib, sy_lib_basic = generate_symbol_library(expr_config["num_variables"], expr_config["symbols"] + extra_symbols,
                                     expr_config["max_arity"], expr_config["has_constants"])
    Node.add_symbols(sy_lib)
    so2 = {s["symbol"]: s for s in sy_lib_basic}

    # Optional (recommended): Generate training set from a custom grammar
    grammar = None

    if grammar is None:
        grammar = generate_grammar(sy_lib)

    f = open("../data/trig15k_newgrammar.txt", 'a')
    trees = read_expressions_json("../data/expression_sets/trig15k_newgrammar.json")
    for t in trees:
        f.write("".join(t.to_list()) + "\n")
    f.close()
    input()


    '''trees = read_expressions_json("../data/expression_sets/ae2k_newgrammar.json")
    exprs = []
    for tree in trees:
        exprs.append(tree.to_list())

    with open("ae2k_newgrammar.pkl", "wb") as f:
        pickle.dump(exprs, f)
    print("Done!")

    input()'''

    lengths = [0] * 42
    heights = [0] * 25
    n_nodes = [0] * 42  # number of nodes in the expression tree
    max_length = -1
    trees = read_expressions_json("../data/expression_sets/trig15k_newgrammar.json")
    for t in trees:
        # print("".join(t.to_list()))
        length = len([i for i in t.to_list() if i not in ["(", ")"]])
        if length > max_length:
            max_length = length
        lengths[length - 1] += 1
        height = t.height()
        heights[height-1] += 1
        n_nodes[len(t) - 1] += 1
    print(lengths)
    print(max_length)
    print(heights)
    print(n_nodes)
    input()

    '''print(tokens_to_tree(["sin", "A", "+", "A", "-", "A", "+", "A"], so2, expr_config["max_arity"]).to_pexpr())
    input()'''

    '''t1 = tokens_to_tree(['(', 'A', '+', 'A', '+', 'A', ')', '*', 'A'], so2, expr_config["max_arity"])
    t2 = tokens_to_tree(['A', '+', 'A', '*', 'A', '+', 'A'], so2, expr_config["max_arity"])

    print(editdistance.eval(t1.to_list("postfix"), t2.to_list("postfix")))
    print(zss.simple_distance(t1, t2, get_label=Node.get_symbol, label_dist=symbol_distance, return_operations=True))'''

    expressions = generate_expressions(grammar, es_config["num_expressions"], so2, expr_config["max_arity"],
                                       max_depth=es_config["max_tree_height"])
    print(len(expressions))

    expr_dict = [tree.to_dict() for tree in expressions]

    save_path = es_config["expression_set_path"]
    if save_path != "":
        with open(save_path, "w") as file:
            json.dump(expr_dict, file)
