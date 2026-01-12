import torch
from torch.autograd import Variable
from symbol_library import SymType


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


class Node:
    _symbols = None
    _s2c = None

    def __init__(self, symbol=None, children=[]):
        self.symbol = symbol
        self.children = children
        self.target = None
        self.prediction = None

    def __str__(self):
        return "".join(self.to_list())

    def __len__(self):
        len_sum = 0
        for t in self.children:
            len_sum += len(t)
        return 1 + len_sum

    def get_symbol(self):
        return self.symbol

    def max_branching_factor(self):
        if len(self.children) == 0:
            return 0
        return max([len(self.children)] + [c.max_branching_factor() for c in self.children])

    def to_pexpr(self):
        self.children = [t for t in self.children if t is not None]
        if Node._symbols is None:
            raise Exception("To generate a pexpr, symbol library is needed. Please use"
                            " the Node.add_symbols methods to add them, before using the to_list method.")
        expression = [Node._symbols[Node._s2c[self.symbol]]["psymbol"]]
        if len(self.children) != 0:
            expression += '('
        for t in self.children:
            expression += t.to_pexpr()
        if len(self.children) != 0:
            expression += ')'
        return expression

    @staticmethod
    def symbol_type(symbol):
        return Node._symbols[Node._s2c[symbol]]["type"].value

    @staticmethod
    def symbol_precedence(symbol):
        return Node._symbols[Node._s2c[symbol]]["precedence"]

    @staticmethod
    def symbol_arity(symbol):
        return Node._symbols[Node._s2c[symbol]]["arity"]

    @staticmethod
    def has_precedence(symbol1, symbol2):
        return (Node.symbol_precedence(symbol1) == Node.symbol_precedence(symbol2)
                and (symbol1 == '-' or symbol1 == '/'))

    def to_list(self):
        self.children = [t for t in self.children if t is not None]
        if Node._symbols is None:
            raise Exception("To generate a list of token in the infix notation, symbol library is needed. Please use"
                            " the Node.add_symbols methods to add them, before using the to_list method.")

        if is_float(self.symbol):
            return [self.symbol]
        stype = Node.symbol_type(self.symbol)
        if stype == SymType.Var.value or stype == SymType.Const.value:
            return [self.symbol]
        elif stype == SymType.Fun.value and len(self.children) == 1:
            expression = self.children[0].to_list()
            if Node.symbol_precedence(self.symbol) > 0:
                return [self.symbol, "("] + expression + [")"]
            else:
                if len(self.children[0]) > 1:
                    expression = ["("] + expression + [")"]
                return expression + [self.symbol]
        elif stype == SymType.Operator.value:
            expression = []
            first = True
            for t in self.children:
                expression += [self.symbol[0]] if not first else []
                subexpression = t.to_list()
                if -1 < Node.symbol_precedence(t.symbol) < Node.symbol_precedence(self.symbol) \
                        or (not first and Node.has_precedence(self.symbol, t.symbol)):
                    subexpression = ["("] + subexpression + [")"]
                expression += subexpression
                first = False
            return expression
        else:
            raise Exception("Invalid symbol type")

    def add_target_vectors(self):
        if Node._symbols is None:
            raise Exception("Encoding needs a symbol library to create target vectors. Please use Node.add_symbols"
                            " method to add a symbol library to trees before encoding.")
        target = torch.zeros(len(Node._symbols)).float()
        target[Node._s2c[self.symbol]] = 1.0
        self.target = Variable(target[None, :])
        for t in self.children:
            t.add_target_vectors()

    def loss(self, mu, logvar, lmbda, criterion):
        pred = Node.to_matrix(self, "prediction")
        target = Node.to_matrix(self, "target")
        BCE = criterion(pred, target)
        KLD = (lmbda * -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))
        return BCE + KLD, BCE, KLD

    def to_dict(self):
        d = {'s': self.symbol}
        i = 1
        for t in self.children:
            d[i] = t.to_dict()
            i += 1
        return d

    @staticmethod
    def from_dict(d):
        children = []
        i = 1
        while str(i) in d:
            children.append(Node.from_dict(d[str(i)]))
            i += 1
        return Node(d["s"], children=children)

    @staticmethod
    def to_matrix(tree, matrix_type="prediction"):
        reps = []

        if matrix_type == "target":
            reps.append(torch.Tensor([Node._s2c[tree.symbol]]).long())
        else:
            reps.append(tree.prediction[0, :])

        for t in tree.children:
            reps.append(Node.to_matrix(t, matrix_type))

        return torch.cat(reps)

    @staticmethod
    def add_symbols(symbols):
        Node._symbols = symbols
        Node._s2c = {s["key"]: i for i, s in enumerate(symbols)}


class BatchedNode:
    _symbols = None
    _s2c = None

    def __init__(self, size=0, trees=None):
        self.symbols = ["" for _ in range(size)]
        self.children = []

        if trees is not None:
            for tree in trees:
                self.add_tree(tree)


    @staticmethod
    def add_symbols(symbols):
        BatchedNode._symbols = symbols
        BatchedNode._s2c = {s["key"]: i for i, s in enumerate(symbols)}

    def add_tree(self, tree=None):
        if tree is None:
            self.symbols.append("")

            for t in self.children:
                t.add_tree()
        else:
            self.symbols.append(tree.symbol)

            s_len, t_len = len(self.children), len(tree.children)
            if s_len > 0 and t_len > 0:
                if s_len <= t_len:
                    for i in range(t_len):
                        if i < s_len:
                            self.children[i].add_tree(tree.children[i])
                        else:
                            child = BatchedNode(size=len(self.symbols)-1)
                            child.add_tree(tree.children[i])
                            self.children.append(child)
                else:
                    for i in range(s_len):
                        if i < t_len:
                            self.children[i].add_tree(tree.children[i])
                        else:
                            self.children[i].add_tree()
            elif s_len > 0:
                for t in self.children:
                    t.add_tree()
            elif t_len > 0:
                self.children = []
                for t in tree.children:
                    child = BatchedNode(size=len(self.symbols)-1)
                    child.add_tree(t)
                    self.children.append(child)

    def loss(self, mu, logvar, lmbda, criterion, beta):
        pred = BatchedNode.get_prediction(self)
        target = BatchedNode.get_target(self)

        pred_f = BatchedNode.get_fraternal_pred(self)
        target_f = BatchedNode.get_fraternal_target(self, [], 0).long()

        BCE = criterion(pred, target) + beta * criterion(pred_f, target_f)
        KLD = (lmbda * -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))/mu.size(0)
        return BCE + KLD, BCE, KLD

    def create_target(self):
        if BatchedNode._symbols is None:
            raise Exception("Encoding needs a symbol library to create target vectors. Please use"
                            " BatchedNode.add_symbols method to add a symbol library to trees before encoding.")
        target = torch.zeros((len(self.symbols), len(Node._symbols)))
        mask = torch.ones(len(self.symbols))

        for i, s in enumerate(self.symbols):
            if s == "":
                mask[i] = 0
            else:
                target[i, Node._s2c[s]] = 1

        self.mask = mask
        self.target = Variable(target)

        for t in self.children:
            t.create_target()

    def to_expr_list(self):
        exprs = []
        for i in range(len(self.symbols)):
            exprs.append(self.get_expr_at_idx(i))
        return exprs

    def get_expr_at_idx(self, idx):
        symbol = self.symbols[idx]
        if symbol == "":
            return None

        exprs = []
        for t in self.children:
            t_expr = t.get_expr_at_idx(idx)
            if t_expr is not None:
                exprs.append(t.get_expr_at_idx(idx))

        return Node(symbol, children=exprs)

    @staticmethod
    def get_prediction(tree):
        reps = []

        reps.append(tree.prediction[:, :, None])

        for t in tree.children:
            reps.append(BatchedNode.get_prediction(t))

        return torch.cat(reps, dim=2)

    @staticmethod
    def get_target(tree):
        reps = []

        target = torch.zeros(len(tree.symbols)).long()
        for i, s in enumerate(tree.symbols):
            if s == "":
                target[i] = -1
            else:
                target[i] = BatchedNode._s2c[s]
        reps.append(target[:, None])

        for t in tree.children:
            reps.append(BatchedNode.get_target(t))

        return torch.cat(reps, dim=1)

    @staticmethod
    def get_fraternal_pred(tree):
        preds = []

        preds.append(tree.fraternal['prediction'][:, :, None])

        i = 0
        for t in tree.children:
            preds.append(BatchedNode.get_fraternal_pred(t))
            i += 1

        return torch.cat(preds, dim=2)

    @staticmethod
    def get_fraternal_target(tree, symbols, num):
        targets = []

        test = torch.zeros(tree.fraternal['target'].size())[:, None]
        for i in range(len(symbols)):
            if symbols[i] not in ['+', '*'] or num == 0: #or num == 4:
                test[i] = -1
            else:
                test[i] = tree.fraternal['target'][i]
        if len(symbols) == 0:
            for i in range(32):
                test[i] = -1
        targets.append(test)

        i = 0
        for t in tree.children:
            targets.append(BatchedNode.get_fraternal_target(t, tree.symbols, i))
            i += 1

        return torch.cat(targets, dim=1)

    @staticmethod
    def get_symbols():
        return BatchedNode._symbols



