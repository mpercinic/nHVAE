import commentjson as cjson
import json
from symbol_library import SymType
from tree import Node, BatchedNode, is_float

def read_trees_json(filepath):
    with open(filepath, "r") as file:
        return [Node.from_dict(d) for d in json.load(file)]


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
    if len(out_stack[-1].to_list('expr')) < num_tokens:
        raise Exception(f"Error while parsing expression {start_expr}")
    return out_stack[-1]


def load_config_file(path):
    with open(path, "r") as file:
        jo = cjson.load(file)
    return jo


def create_batch(trees):
    t = BatchedNode(trees=trees)
    t.create_target()
    return t

all_symbols_neuro = {
"Node": {"key": 'Node', "min_arity": 4, "max_arity": 10},
"Pops2": {"key": 'Pops2', "min_arity": 1, "max_arity": 2},
"Pop1": {"key": 'Pop1', "arity": 5},
"Pop2": {"key": 'Pop2', "arity": 8},
"InputDyn": {"key": 'InputDyn', "min_arity": 1, "max_arity": 2},
"OutputDyn": {"key": 'OutputDyn', "arity": 1},
"ECF": {"key": 'ECF', "arity": 1},
"SCF": {"key": 'SCF', "arity": 1},
"CF": {"key": 'CF', "arity": 1},
"S": {"key": 'S', "min_arity": 1, "max_arity": 3},
"P": {"key": 'P', "min_arity": 1, "max_arity": 3},
"V": {"key": 'V', "arity": 1},
"CM1": {"key": 'CM1', "arity": 1},
"CM2": {"key": 'CM2', "arity": 1},
"CM": {"key": 'CM', "min_arity": 1, "max_arity": 3},
"Digit": {"key": 'Digit', "arity": 1},
"SN": {"key": 'SN', "arity": 1},
"second_order_kernel": {"key": 'second_order_kernel', "arity": 0},
"exp_kernel": {"key": 'exp_kernel', "arity": 0},
"gating_kinetics": {"key": 'gating_kinetics', "arity": 0},
"linear_kernel": {"key": 'linear_kernel', "arity": 0},
"voltage_gated_dynamics": {"key": 'voltage_gated_dynamics', "arity": 0},
"direct_readout": {"key": 'direct_readout', "arity": 0},
"membrane_integrator": {"key": 'membrane_integrator', "arity": 0},
"difference": {"key": 'difference', "arity": 0},
"spatial_gradient": {"key": 'spatial_gradient', "arity": 0},
"linear": {"key": 'linear', "arity": 0},
"custom": {"key": 'custom', "arity": 0},
"false": {"key": 'false', "arity": 0},
"saturating_sigmoid": {"key": 'saturating_sigmoid', "arity": 0},
"relaxed_rectifier": {"key": 'relaxed_rectifier', "arity": 0},
"baseline_sigmoid": {"key": 'baseline_sigmoid', "arity": 0},
"+": {"key": '+', "arity": 0},
"*": {"key": '*', "arity": 0},
"x1": {"key": 'x1', "arity": 0},
"x2": {"key": 'x2', "arity": 0},
"full": {"key": 'full', "arity": 0},
"null": {"key": 'null', "arity": 0},
"ring": {"key": 'ring', "arity": 0},
"star": {"key": 'star', "arity": 0},
"hub_tail": {"key": 'hub_tail', "arity": 0},
"star_feedback_tail": {"key": 'star_feedback_tail', "arity": 0},
"star_loop_extended": {"key": 'star_loop_extended', "arity": 0},
"ei_extended": {"key": 'ei_extended', "arity": 0},
"small_world": {"key": 'small_world', "arity": 0},
"0": {"key": '0', "arity": 0},
"1": {"key": '1', "arity": 0},
"2": {"key": '2', "arity": 0},
"3": {"key": '3', "arity": 0},
"4": {"key": '4', "arity": 0},
"5": {"key": '5', "arity": 0},
"6": {"key": '6', "arity": 0},
"7": {"key": '7', "arity": 0},
"8": {"key": '8', "arity": 0},
"9": {"key": '9', "arity": 0},
"true": {"key": 'true', "arity": 0}
}
