import numpy as np
from typing import Union, Any
from collections import defaultdict

def constraint_Node(grammar, rule, constraints):
    constraints["num_nodes"] = len(grammar.rules[rule].rhs)
    return constraints


def constraint_NumIncomingConns(grammar, constraints):
    if "num_nodes" not in constraints:
        return grammar.rule_masks["NumIncomingConns"], grammar.rule_probs["NumIncomingConns"], constraints
    else:
        applicable_rules = grammar.rule_masks["NumIncomingConns"]
        applicable_rule_probs = grammar.rule_probs["NumIncomingConns"]
        potential_rules = []
        potential_rule_probs = []
        total_prob = 0
        for rule, prob in zip(applicable_rules, applicable_rule_probs):
            if int(grammar.rules[rule].rhs[0]) <= constraints["num_nodes"] or constraints["num_nodes"] >= 3:
                potential_rules.append(rule)
                potential_rule_probs.append(prob)
                total_prob += prob
        potential_rules_probs = [prob / total_prob for prob in potential_rule_probs]
        return np.array(potential_rules), np.array(potential_rules_probs), constraints


def constraint_ConnFuncMode(grammar, constraints):
    if "num_nodes" in constraints and constraints["num_nodes"] < 3:
        return np.array([grammar.rhs_to_rule("ConnFuncMode", ["per_source"])]), np.array([1.0]), constraints
    else:
        return grammar.rule_masks["ConnFuncMode"], grammar.rule_probs["ConnFuncMode"], constraints

class Rule:
    num_rules = 0

    def __init__(self, lhs: str, rhs: list[str], weight: float = 1.0):
        self.id = Rule.num_rules
        Rule.num_rules += 1
        self.lhs = lhs
        self.rhs = rhs
        self.weight = weight
        self.nt_rhs = None

    def __str__(self):
        return f"{self.lhs} -> {' '.join(self.rhs)}"

    def add_nt_rhs(self, non_terminals):
        self.nt_rhs = [s for s in self.rhs if s in non_terminals]

class Grammar:
    def __init__(self, grammar: Union[list[Rule], str], starting_symbol: str, constraints: dict[str, Any]=None):
        # Transform the grammar in the string form into the rules object if needed
        if isinstance(grammar, str):
            rules = []
            for line in grammar.strip().split('\n'):
                non_terminal, productions = line.split(' -> ')
                productions = productions.split(' | ')
                for production in productions:
                    if production.strip()[-1] == ']':
                        probability = float(production.split('[')[1].split(']')[0])
                        production = production.split('[')[0].strip()
                    else:
                        probability = 1.0
                    production = production.replace('"', "")
                    rules.append(Rule(non_terminal, production.strip().split(), probability))
        else:
            rules = grammar

        self.rules = rules
        self.starting_symbol = starting_symbol
        self.constraints = constraints
        self.non_terminals = set()
        self.all_symbols = set()
        total_weights = defaultdict(lambda: 0)
        for rule in rules:
            total_weights[rule.lhs] += rule.weight
            self.non_terminals.add(rule.lhs)
            for symbol in rule.rhs:
                self.all_symbols.add(symbol)
        self.terminals = self.all_symbols - self.non_terminals

        for rule in rules:
            rule.add_nt_rhs(self.non_terminals)

        self.rule_masks = dict()
        self.rule_probs = dict()
        for non_terminal in self.non_terminals:
            self.rule_masks[non_terminal] = list()
            self.rule_probs[non_terminal] = list()
            for rule in rules:
                if rule.lhs == non_terminal:
                    self.rule_masks[non_terminal].append(rule.id)
                    self.rule_probs[non_terminal].append(rule.weight/total_weights[non_terminal])
        self.all_symbols = list(self.all_symbols)

    def print_rules(self):
        for rule in self.rules:
            print(rule)

    def print_rule_masks(self):
        for non_terminal in self.non_terminals:
            print(f"{non_terminal} -> {self.rule_masks[non_terminal]}")

    def rhs_in_mask(self, non_terminal, rhs):
        return rhs in [self.rules[index].rhs for index in self.rule_masks[non_terminal]]

    def rhs_to_rule(self, non_terminal, rhs):
        possible_rules = self.rule_masks[non_terminal]
        for rule in possible_rules:
            if self.rules[rule].rhs == rhs:
                return rule
        raise Exception("Rule not found in mask!")

    def max_arity(self):
        return max([len(rule.nt_rhs) for rule in self.rules])

    def select_possible_rules(self, symbol, constraints=None):
        if symbol not in self.non_terminals:
            # Shouldn't happen?
            return np.array([]), np.array([]), constraints

        if constraints is None:
            return self.rule_masks[symbol], self.rule_probs[symbol], None
        else:
            if symbol+"_pre" in constraints:
                rules, probabilities, constraints = constraints[symbol+"_pre"](self, constraints)
            else:
                rules, probabilities, constraints = self.rule_masks[symbol], self.rule_probs[symbol], constraints

            if "any_pre" in constraints:
                return constraints["any_pre"](self, symbol, rules, probabilities, constraints)
            else:
                return rules, probabilities, constraints

    def update_constraints(self, rule, constraints=None):
        symbol = self.rules[rule].lhs
        if constraints is None:
            return constraints
        if symbol+"_post" in constraints:
            constraints = constraints[symbol+"_post"](self, rule, constraints)
        if "any_post" in constraints:
            constraints = constraints["any_post"](self, symbol, rule, constraints)
        return constraints