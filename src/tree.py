import torch
from torch.autograd import Variable
from symbol_library import SymType


def is_float(element: any) -> bool:
    #If you expect None to be passed:
    if element is None:
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

    def height(self):
        height_max = 0
        for t in self.children:
            height_max = t.height()
        return 1 + height_max

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

    def to_list(self, notation="infix"):
        self.children = [t for t in self.children if t is not None]
        if notation == "infix" and Node._symbols is None:
            raise Exception("To generate a list of token in the infix notation, symbol library is needed. Please use"
                            " the Node.add_symbols methods to add them, before using the to_list method.")
        if notation == "prefix":
            expression = [self.symbol]
            for t in self.children:
                expression += t.to_list(notation)
            return expression
        elif notation == "postfix":
            expression = []
            for t in self.children:
                expression += t.to_list(notation)
            return expression + [self.symbol]

        elif notation == "infix":
            # if is_float(self.symbol):
                # return [self.symbol]
            stype = Node.symbol_type(self.symbol)
            if stype == SymType.Var.value or stype == SymType.Const.value:
                return [self.symbol]
            # len(self.children) == 1 redundant?
            elif stype == SymType.Fun.value and len(self.children) == 1:
                expression = self.children[0].to_list(notation)
                if Node.symbol_precedence(self.symbol) > 0:
                    return [self.symbol, "("] + expression + [")"]
                else:
                    if len(self.children[0]) > 1:
                        expression = ["("] + expression + [")"]
                    return expression + [self.symbol]
            # elif stype == SymType.Fun.value:
                # print(":" + self.symbol, self.children)
            elif stype == SymType.Operator.value:
                # print(self.symbol, self.children)
                expression = []
                first = True
                for t in self.children:
                    expression += [self.symbol[0]] if not first else []
                    subexpression = t.to_list(notation)
                    if -1 < Node.symbol_precedence(t.symbol) < Node.symbol_precedence(self.symbol) \
                            or (not first and Node.has_precedence(self.symbol, t.symbol)):
                        subexpression = ["("] + subexpression + [")"]
                    expression += subexpression
                    first = False
                return expression
            else:
                '''print(self.symbol)
                for c in self.children:
                    print(c.symbol)'''
                raise Exception("Invalid symbol type")
        else:
            raise Exception("Invalid notation selected. Use 'infix', 'prefix', 'postfix'.")

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

    def add_target_vectors(self):
        if Node._symbols is None:
            raise Exception("Encoding needs a symbol library to create target vectors. Please use Node.add_symbols"
                            " method to add a symbol library to trees before encoding.")
        target = torch.zeros(len(Node._symbols)).float()
        target[Node._s2c[self.symbol]] = 1.0
        self.target = Variable(target[None, None, :])
        for t in self.children:
            t.add_target_vectors()

    def loss(self, mu, logvar, lmbda, criterion):
        pred = Node.to_matrix(self, "prediction")
        target = Node.to_matrix(self, "target")
        BCE = criterion(pred, target)
        KLD = (lmbda * -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))
        return BCE + KLD, BCE, KLD

    def clear_prediction(self):
        for t in self.children:
            t.clear_prediction()
        self.prediction = None

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
            reps.append(tree.prediction[0, :, :])

        for t in tree.children:
            reps.append(Node.to_matrix(t, matrix_type))

        return torch.cat(reps)

    @staticmethod
    def add_symbols(symbols):
        Node._symbols = symbols
        Node._s2c = {s["key"]: i for i, s in enumerate(symbols)}


class BatchedNode():
    _symbols = None
    _s2c = None

    def __init__(self, size=0, trees=None):
        self.symbols = ["" for _ in range(size)]
        #self.left = None
        #self.right = None
        self.children = []

        if trees is not None:
            for tree in trees:
                self.add_tree(tree)


    @staticmethod
    def add_symbols(symbols):
        BatchedNode._symbols = symbols
        BatchedNode._s2c = {s["key"]: i for i, s in enumerate(symbols)}

    def batched_len(self):
        symbol_length = 0
        for s in self.symbols:
            if len(s) > 0:
                symbol_length += 1
        len_sum = 0.0
        for t in self.children:
            len_sum += t.batched_len()
        return symbol_length / len(self.symbols) + len_sum

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

    def loss(self, mu, logvar, lmbda, criterion):
        pred = BatchedNode.get_prediction(self)
        pred = torch.permute(pred, [0, 2, 1])
        target = BatchedNode.get_target(self)
        BCE = criterion(pred, target)
        KLD = (lmbda * -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))/mu.size(0)
        return BCE + KLD, BCE, KLD

    def create_target(self):
        if BatchedNode._symbols is None:
            raise Exception("Encoding needs a symbol library to create target vectors. Please use"
                            " BatchedNode.add_symbols method to add a symbol library to trees before encoding.")
        target = torch.zeros((len(self.symbols), 1, len(Node._symbols)))
        mask = torch.ones(len(self.symbols))

        for i, s in enumerate(self.symbols):
            if s == "":
                mask[i] = 0
            else:
                target[i, 0, Node._s2c[s]] = 1

        self.mask = mask
        self.target = Variable(target)

        for t in self.children:
            t.create_target()

        '''if self.left is not None:
            self.left.create_target()
        if self.right is not None:
            self.right.create_target()'''

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
            exprs.append(t.get_expr_at_idx(idx))

        #left = self.left.get_expr_at_idx(idx) if self.left is not None else None
        #right = self.right.get_expr_at_idx(idx) if self.right is not None else None

        return Node(symbol, children=exprs)

    @staticmethod
    def get_prediction(tree):
        reps = []
        '''if tree.left is not None:
            reps.append(BatchedNode.get_prediction(tree.left))'''

        '''p = []
        for i in range(len(BatchedNode._symbols)):
            p += [(BatchedNode._symbols[i]["key"], i+1)]
        print(p)
        print(tree.symbols)
        print(tree.prediction)'''

        target = tree.prediction[:, 0, :]
        reps.append(target[:, None, :])

        '''if tree.right is not None:
            reps.append(BatchedNode.get_prediction(tree.right))'''

        for t in tree.children:
            reps.append(BatchedNode.get_prediction(t))

        return torch.cat(reps, dim=1)

    @staticmethod
    def get_target(tree):
        reps = []
        '''if tree.left is not None:
            reps.append(BatchedNode.get_target(tree.left))'''

        target = torch.zeros(len(tree.symbols)).long()
        for i, s in enumerate(tree.symbols):
            if s == "":
                target[i] = -1
            else:
                target[i] = BatchedNode._s2c[s]
        reps.append(target[:, None])

        for t in tree.children:
            reps.append(BatchedNode.get_target(t))

        '''if tree.right is not None:
            reps.append(BatchedNode.get_target(tree.right))'''

        return torch.cat(reps, dim=1)



