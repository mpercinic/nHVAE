import commentjson as cjson
import json
from symbol_library import SymType
from tree import Node, BatchedNode, is_float

def read_expressions_json(filepath):
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
    if len(out_stack[-1].to_list()) < num_tokens:
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
