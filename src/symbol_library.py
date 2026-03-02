from enum import Enum

class SymType(Enum):
    Var = 1
    Const = 2
    Operator = 3
    Fun = 4
    Literal = 5

def generate_symbol_library(num_vars, symbol_list, dataset, has_constant=True):
    if dataset == "expr":
        all_symbols = {
            "+": {"symbol": '+', "type": SymType.Operator, "precedence": 0, "psymbol": "add", "key": "+", "min_arity": 2},
            "-": {"symbol": '-', "type": SymType.Operator, "precedence": 0, "psymbol": "sub", "arity": 2, "key": "-"},
            "*": {"symbol": '*', "type": SymType.Operator, "precedence": 1, "psymbol": "mul", "key": "*", "min_arity": 2},
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
    elif dataset == "neuro":
        all_symbols = {
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
    symbols = []
    for i in range(num_vars):
        symbols.append({"symbol": 'X_'+str(i), "type": SymType.Var, "precedence": 5, "psymbol": 'X_'+str(i),
                        "arity": 0, "key": 'X_'+str(i)})

    if has_constant:
        symbols.append({"symbol": 'C', "type": SymType.Const, "precedence": 5, "psymbol": "const", "arity": 0, "key": 'C'})

    for s in symbol_list:
        if s in all_symbols:
            symbols.append(all_symbols[s])
        else:
            raise Exception(f"Symbol {s} is not in the standard library, please add it into the all_symbols variable"
                            f" from the generate_symbol_library method in symbol_library.py")

    return symbols
