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
            "Node": {"key": 'Node', "min_arity": 4, "max_arity": 10, "psymbol": "Node"},
            "Pops2": {"key": 'Pops2', "min_arity": 1, "max_arity": 2, "psymbol": "Pops2"},
            "Pop1": {"key": 'Pop1', "arity": 5, "psymbol": "Pop1"},
            "Pop2": {"key": 'Pop2', "arity": 8, "psymbol": "Pop2"},
            "InputDyn": {"key": 'InputDyn', "min_arity": 1, "max_arity": 2, "psymbol": "InputDyn"},
            "OutputDyn": {"key": 'OutputDyn', "arity": 1, "psymbol": "OutputDyn"},
            "ECF": {"key": 'ECF', "arity": 1, "psymbol": "ECF"},
            "SCF": {"key": 'SCF', "arity": 1, "psymbol": "SCF"},
            "CF": {"key": 'CF', "arity": 1, "psymbol": "CF"},
            "S": {"key": 'S', "min_arity": 1, "max_arity": 3, "psymbol": "S"},
            "P": {"key": 'P', "min_arity": 1, "max_arity": 3, "psymbol": "P"},
            "V": {"key": 'V', "arity": 1, "psymbol": "V"},
            "CM1": {"key": 'CM1', "arity": 1, "psymbol": "CM1"},
            "CM2": {"key": 'CM2', "arity": 1, "psymbol": "CM2"},
            "CM": {"key": 'CM', "min_arity": 1, "max_arity": 3, "psymbol": "CM"},
            "Digit": {"key": 'Digit', "arity": 1, "psymbol": "Digit"},
            "SN": {"key": 'SN', "arity": 1, "psymbol": "SN"},
            "second_order_kernel": {"key": 'second_order_kernel', "arity": 0, "psymbol": "second_order_kernel"},
            "exp_kernel": {"key": 'exp_kernel', "arity": 0, "psymbol": "exp_kernel"},
            "gating_kinetics": {"key": 'gating_kinetics', "arity": 0, "psymbol": "gating_kinetics"},
            "linear_kernel": {"key": 'linear_kernel', "arity": 0, "psymbol": "linear_kernel"},
            "voltage_gated_dynamics": {"key": 'voltage_gated_dynamics', "arity": 0, "psymbol": "voltage_gated_dynamics"},
            "direct_readout": {"key": 'direct_readout', "arity": 0, "psymbol": "direct_readout"},
            "membrane_integrator": {"key": 'membrane_integrator', "arity": 0, "psymbol": "membrane_integrator"},
            "difference": {"key": 'difference', "arity": 0, "psymbol": "difference"},
            "spatial_gradient": {"key": 'spatial_gradient', "arity": 0, "psymbol": "spatial_gradient"},
            "linear": {"key": 'linear', "arity": 0, "psymbol": "linear"},
            "custom": {"key": 'custom', "arity": 0, "psymbol": "custom"},
            "false": {"key": 'false', "arity": 0, "psymbol": "false"},
            "saturating_sigmoid": {"key": 'saturating_sigmoid', "arity": 0, "psymbol": "saturating_sigmoid"},
            "relaxed_rectifier": {"key": 'relaxed_rectifier', "arity": 0, "psymbol": "relaxed_rectifier"},
            "baseline_sigmoid": {"key": 'baseline_sigmoid', "arity": 0, "psymbol": "baseline_sigmoid"},
            "+": {"key": '+', "arity": 0, "psymbol": "+"},
            "*": {"key": '*', "arity": 0, "psymbol": "*"},
            "x1": {"key": 'x1', "arity": 0, "psymbol": "x1"},
            "x2": {"key": 'x2', "arity": 0, "psymbol": "x2"},
            "full": {"key": 'full', "arity": 0, "psymbol": "full"},
            "null": {"key": 'null', "arity": 0, "psymbol": "null"},
            "ring": {"key": 'ring', "arity": 0, "psymbol": "ring"},
            "star": {"key": 'star', "arity": 0, "psymbol": "star"},
            "hub_tail": {"key": 'hub_tail', "arity": 0, "psymbol": "hub_tail"},
            "star_feedback_tail": {"key": 'star_feedback_tail', "arity": 0, "psymbol": "star_feedback_tail"},
            "star_loop_extended": {"key": 'star_loop_extended', "arity": 0, "psymbol": "star_loop_extended"},
            "ei_extended": {"key": 'ei_extended', "arity": 0, "psymbol": "ei_extended"},
            "small_world": {"key": 'small_world', "arity": 0, "psymbol": "small_world"},
            "0": {"key": '0', "arity": 0, "psymbol": "0"},
            "1": {"key": '1', "arity": 0, "psymbol": "1"},
            "2": {"key": '2', "arity": 0, "psymbol": "2"},
            "3": {"key": '3', "arity": 0, "psymbol": "3"},
            "4": {"key": '4', "arity": 0, "psymbol": "4"},
            "5": {"key": '5', "arity": 0, "psymbol": "5"},
            "6": {"key": '6', "arity": 0, "psymbol": "6"},
            "7": {"key": '7', "arity": 0, "psymbol": "7"},
            "8": {"key": '8', "arity": 0, "psymbol": "8"},
            "9": {"key": '9', "arity": 0, "psymbol": "9"},
            "true": {"key": 'true', "arity": 0, "psymbol": "true"}
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
