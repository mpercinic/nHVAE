from enum import Enum


class SymType(Enum):
    Var = 1
    Const = 2
    Operator = 3
    Fun = 4
    Literal = 5


def generate_symbol_library(num_vars, symbol_list, max_arity, has_constant=True):
    all_symbols = {
        # "+": {"symbol": '+', "type": SymType.Operator, "precedence": 0, "psymbol": "add"},
        "-": {"symbol": '-', "type": SymType.Operator, "precedence": 0, "psymbol": "sub", "arity": 2, "key": "-"},
        # "*": {"symbol": '*', "type": SymType.Operator, "precedence": 1, "psymbol": "mul"},
        "/": {"symbol": '/', "type": SymType.Operator, "precedence": 1, "psymbol": "div", "arity": 2, "key": "/"},
        "^": {"symbol": "^", "type": SymType.Operator, "precedence": 2, "psymbol": "pow", "arity": 2, "key": "^"},
        "sqrt": {"symbol": 'sqrt', "type": SymType.Fun, "precedence": 5, "psymbol": "sqrt", "arity": 1, "key": "sqrt"},
        "sin": {"symbol": 'sin', "type": SymType.Fun, "precedence": 5, "psymbol": "sin", "arity": 1, "key": "sin"},
        "cos": {"symbol": 'cos', "type": SymType.Fun, "precedence": 5, "psymbol": "cos", "arity": 1, "key": "cos"},
        "exp": {"symbol": 'exp', "type": SymType.Fun, "precedence": 5, "psymbol": "exp", "arity": 1, "key": "exp"},
        "log": {"symbol": 'log', "type": SymType.Fun, "precedence": 5, "psymbol": "log", "arity": 1, "key": "log"},
        "^2": {"symbol": '^2', "type": SymType.Fun, "precedence": -1, "psymbol": "n2", "arity": 1, "key": "^2"},
        "^3": {"symbol": '^3', "type": SymType.Fun, "precedence": -1, "psymbol": "n3", "arity": 1, "key": "^3"},
        "^4": {"symbol": '^4', "type": SymType.Fun, "precedence": -1, "psymbol": "n4", "arity": 1, "key": "^4"},
        "^5": {"symbol": '^5', "type": SymType.Fun, "precedence": -1, "psymbol": "n5", "arity": 1, "key": "^5"},
    }
    for i in range(2, max_arity + 1):
        all_symbols["+" + str(i)] = {"symbol": '+', "type": SymType.Operator, "precedence": 0, "psymbol": "add",
                                     "arity": i, "key": "+" + str(i)}
        all_symbols["*" + str(i)] = {"symbol": '*', "type": SymType.Operator, "precedence": 1, "psymbol": "mul",
                                     "arity": i, "key": "*" + str(i)}
        '''all_symbols["-" + str(i)] = {"symbol": '-', "type": SymType.Operator, "precedence": 0, "psymbol": "sub",
                                     "arity": i, "key": "-" + str(i)}
        all_symbols["/" + str(i)] = {"symbol": '/', "type": SymType.Operator, "precedence": 1, "psymbol": "div",
                                     "arity": i, "key": "/" + str(i)}'''
    variable_names = 'ABDEFGHIJKLMNOPQRSTUVWXYZČŠŽ'
    symbols = []
    for i in range(num_vars):
        if i < len(variable_names):
            symbols.append({"symbol": variable_names[i], "type": SymType.Var, "precedence": 5, "psymbol": variable_names[i],
                            "arity": 0, "key": variable_names[i]})
        else:
            raise Exception("Insufficient symbol names, please add additional symbols into the variable_names variable"
                            " from the generate_symbol_library method in symbol_library.py")

    if has_constant:
        symbols.append({"symbol": 'C', "type": SymType.Const, "precedence": 5, "psymbol": "const", "arity": 0, "key": 'C'})

    for s in symbol_list:
        if s in all_symbols:
            symbols.append(all_symbols[s])
        else:
            raise Exception(f"Symbol {s} is not in the standard library, please add it into the all_symbols variable"
                            f" from the generate_symbol_library method in symbol_library.py")

    all_symbols2 = {
        "+": {"symbol": '+', "type": SymType.Operator, "precedence": 0, "psymbol": "add"},
        "-": {"symbol": '-', "type": SymType.Operator, "precedence": 0, "psymbol": "sub", "arity": 2},
        "*": {"symbol": '*', "type": SymType.Operator, "precedence": 1, "psymbol": "mul"},
        "/": {"symbol": '/', "type": SymType.Operator, "precedence": 1, "psymbol": "div", "arity": 2},
        "^": {"symbol": "^", "type": SymType.Operator, "precedence": 2, "psymbol": "pow", "arity": 2},
        "sqrt": {"symbol": 'sqrt', "type": SymType.Fun, "precedence": 5, "psymbol": "sqrt", "arity": 1},
        "sin": {"symbol": 'sin', "type": SymType.Fun, "precedence": 5, "psymbol": "sin", "arity": 1},
        "cos": {"symbol": 'cos', "type": SymType.Fun, "precedence": 5, "psymbol": "cos", "arity": 1},
        "exp": {"symbol": 'exp', "type": SymType.Fun, "precedence": 5, "psymbol": "exp", "arity": 1},
        "log": {"symbol": 'log', "type": SymType.Fun, "precedence": 5, "psymbol": "log", "arity": 1},
        "^2": {"symbol": '^2', "type": SymType.Fun, "precedence": -1, "psymbol": "n2", "arity": 1},
        "^3": {"symbol": '^3', "type": SymType.Fun, "precedence": -1, "psymbol": "n3", "arity": 1},
        "^4": {"symbol": '^4', "type": SymType.Fun, "precedence": -1, "psymbol": "n4", "arity": 1},
        "^5": {"symbol": '^5', "type": SymType.Fun, "precedence": -1, "psymbol": "n5", "arity": 1},
    }
    symbols2 = []
    for i in range(num_vars):
        if i < len(variable_names):
            symbols2.append(
                {"symbol": variable_names[i], "type": SymType.Var, "precedence": 5, "psymbol": variable_names[i],
                 "arity": 0})
        else:
            raise Exception("Insufficient symbol names, please add additional symbols into the variable_names variable"
                            " from the generate_symbol_library method in symbol_library.py")
    if has_constant:
        symbols2.append({"symbol": 'C', "type": SymType.Const, "precedence": 5, "psymbol": "const", "arity": 0})
    for s in all_symbols2:
        symbols2.append(all_symbols2[s])

    return symbols, symbols2
