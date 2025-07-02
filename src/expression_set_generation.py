from argparse import ArgumentParser
import json

from ProGED.generators.grammar import GeneratorGrammar

from tree import Node
from hvae_utils import load_config_file
from symbol_library import generate_symbol_library, SymType


def generate_grammar(symbols):
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
    if len(out_stack[-1].to_list()) < num_tokens:
        raise Exception(f"Could not parse the whole expression {start_expr}")
    return out_stack[-1]


def generate_expressions(grammar, number_of_all_expressions, symbols, max_arity, max_length):
    generator = GeneratorGrammar(grammar)
    expression_set = set()
    expression_trees = []

    while len(expression_set) < number_of_all_expressions:
        if len(expression_set) % 500 == 0:
            print(f"Unique expressions generated so far: {len(expression_set)}")
            if len(expression_trees) > 0:
                print("".join(expression_trees[-1].to_list()))
        expr = generator.generate_one()[0]

        try:
            expr_tree = tokens_to_tree(expr, symbols, max_arity)
            expr_str = "".join(expr_tree.to_list())
            if expr_str in expression_set:
                continue
            if len([s for s in expr_tree.to_list() if s not in ["(", ")"]]) > max_length:
                continue
        except:
            continue
        expression_trees.append(expr_tree)
        expression_set.add(expr_str)

    return expression_trees

def symbol_distance(a, b):
    return int(a != b)


if __name__ == '__main__':
    parser = ArgumentParser(prog='Expression set generation', description='Generate a set of expressions')
    parser.add_argument("-config", default="../configs/config.json")
    args = parser.parse_args()

    config = load_config_file(args.config)
    expr_config = config["expression_definition"]
    es_config = config["expression_set_generation"]

    sy_lib = generate_symbol_library(expr_config["num_variables"], expr_config["symbols"], expr_config["has_constants"])
    Node.add_symbols(sy_lib)
    so = {s["symbol"]: s for s in sy_lib}

    # Optional (recommended): Generate training set from a custom grammar
    grammar = None

    if grammar is None:
        grammar = generate_grammar(sy_lib)

    expressions = generate_expressions(grammar, es_config["num_expressions"], so, expr_config["max_arity"], es_config["max_length"])
    print("Number of expressions generated: " + str(len(expressions)))

    expr_dict = [tree.to_dict() for tree in expressions]

    save_path = es_config["expression_set_path"]
    if save_path != "":
        with open(save_path, "w") as file:
            json.dump(expr_dict, file)
