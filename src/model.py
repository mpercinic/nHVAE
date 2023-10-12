import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from random import random

from tree import Node, BatchedNode
from symbol_library import SymType


class HVAE(nn.Module):
    _symbols = None

    def __init__(self, input_size, output_size, hidden_size=None):
        super(HVAE, self).__init__()

        if hidden_size is None:
            hidden_size = output_size

        self.encoder = Encoder(input_size, hidden_size, output_size)
        self.decoder = Decoder(output_size, hidden_size, input_size)

    def forward(self, tree):
        mu, logvar = self.encoder(tree)
        z = self.sample(mu, logvar)
        out = self.decoder(z, tree)
        return mu, logvar, out

    def sample(self, mu, logvar):
        eps = Variable(torch.randn(mu.size()))
        std = torch.exp(logvar / 2.0)
        return mu + eps * std

    def encode(self, tree):
        mu, logvar = self.encoder(tree)
        return mu, logvar

    def decode(self, z):
        if HVAE._symbols is None:
            raise Exception("To generate expression trees, a symbol library is needed. Please add it using the"
                            " HVAE.add_symbols method.")
        return self.decoder.decode(z, HVAE._symbols)

    @staticmethod
    def add_symbols(symbols):
        HVAE._symbols = symbols
        Node.add_symbols(symbols)
        BatchedNode.add_symbols(symbols)


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.gru = GRU221(input_size=input_size, hidden_size=hidden_size)
        '''self.input_size = input_size
        self.hidden_size = hidden_size
        self.grus = {}'''
        self.mu = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.logvar = nn.Linear(in_features=hidden_size, out_features=output_size)

        torch.nn.init.xavier_uniform_(self.mu.weight)
        torch.nn.init.xavier_uniform_(self.logvar.weight)

    def forward(self, tree):
        # start = time.time()
        # Check if the tree has target vectors
        if tree.target is None:
            tree.add_target_vectors()

        tree_encoding = self.recursive_forward(tree)
        mu = self.mu(tree_encoding)
        logvar = self.logvar(tree_encoding)
        # end = time.time()
        # print("Encoder: " + str(end-start))
        return mu, logvar

    def recursive_forward(self, tree):
        children = []
        #lengths = []
        for t in tree.children:
            children.append(self.recursive_forward(t))
            #lengths.append(t.batched_len())
        if len(children) == 0:
            children.append(torch.zeros(tree.target.size(0), 1, self.hidden_size))
            child_sum = children[0]
        else:
            # l = sum(lengths)
            # child_sum = (lengths[0] / l) * children[0]
            child_sum = sum(children)
            '''for i in range(len(children)):
                if i != 0:
                    child_sum += (lengths[i] / l) * children[i]'''

        # left = left.mul(tree.mask[:, None, None])
        # right = right.mul(tree.mask[:, None, None])

        hidden = self.gru(tree.target, child_sum, children)
        hidden = hidden.mul(tree.mask[:, None, None])
        return hidden


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.z2h = nn.Linear(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.gru_ancestral = GRUAncestral(input_size=output_size, hidden_size=hidden_size)
        self.gru_fraternal = GRUFraternal(input_size=output_size, hidden_size=hidden_size)
        self.u_a = nn.Linear(hidden_size, hidden_size)
        self.u_f = nn.Linear(hidden_size, hidden_size)

        torch.nn.init.xavier_uniform_(self.z2h.weight)
        torch.nn.init.xavier_uniform_(self.h2o.weight)
        torch.nn.init.xavier_uniform_(self.u_a.weight)
        torch.nn.init.xavier_uniform_(self.u_f.weight)

    # Used during training to guide the learning process
    def forward(self, z, tree):
        # start = time.time()
        hidden = self.z2h(z)
        self.recursive_forward(hidden, hidden, tree)
        # end = time.time()
        # print("Decoder: " + str(end - start))
        return tree

    def recursive_forward(self, hidden_a, hidden, tree):
        prediction = self.h2o(hidden)
        symbol_probs = F.softmax(prediction, dim=2)
        tree.prediction = prediction

        hidden_a_i = self.gru_ancestral(symbol_probs, hidden_a)
        first = True
        # hiddens = []
        for t in tree.children:
            if t is not None:
                if first:
                    # hiddens.append(hidden_a_i)
                    symbol_probs_f = self.recursive_forward(hidden_a_i, hidden_a_i, t)
                    # z = torch.zeros(hidden_a.size())
                    # hidden_f = self.z2h(z)
                    # hidden_f = hidden_a_i
                    hidden_f = torch.zeros(hidden_a.size())
                else:
                    hidden_f = self.gru_fraternal(symbol_probs_f, hidden_f)
                    hidden = torch.tanh(self.u_f(hidden_f) + self.u_a(hidden_a_i))
                    # hiddens.append(hidden)
                    symbol_probs_f = self.recursive_forward(hidden_a_i, hidden, t)
                first = False

        # print((tree.symbols, hiddens))
        return symbol_probs

    # Used for inference to generate expression trees from latent vectorS
    def decode(self, z, symbol_dict):
        with torch.no_grad():
            mask = torch.ones(z.size(0)).bool()
            hidden = self.z2h(z)
            batch, _ = self.recursive_decode(hidden, hidden, symbol_dict, mask)
            return batch.to_expr_list()

    def recursive_decode(self, hidden_a, hidden, symbol_dict, mask):
        prediction = F.softmax(self.h2o(hidden), dim=2)
        # Sample symbol in a given node
        symbols, child_mask = self.sample_symbol(prediction, symbol_dict, mask)

        first = True
        hidden_a_i = self.gru_ancestral(prediction, hidden_a)
        children = []
        # hiddens = []
        for i in range(child_mask.size(0)):
            if first:
                # hidden_f = hidden_a_i
                # z = torch.zeros(hidden_a.size())
                # hidden_f = self.z2h(z)
                hidden_f = torch.zeros(hidden_a.size())
                if torch.any(child_mask[i]):
                    # hiddens.append(hidden_a_i)
                    child, prediction_f = self.recursive_decode(hidden_a_i, hidden_a_i, symbol_dict, child_mask[i])
                    children.append(child)
            elif torch.any(child_mask[i]):
                # prediction_f = F.softmax(self.h2o(hidden_f), dim=2)
                hidden_f = self.gru_fraternal(prediction_f, hidden_f)
                hidden = torch.tanh(self.u_f(hidden_f) + self.u_a(hidden_a_i))
                # hiddens.append(hidden)
                child, prediction_f = self.recursive_decode(hidden_a_i, hidden, symbol_dict, child_mask[i])
                children.append(child)
                # print(child_mask)
            first = False

        # print((symbols, hiddens))

        node = BatchedNode()
        node.symbols = symbols
        node.children = children
        return (node, prediction)

    def sample_symbol(self, prediction, symbol_dict, mask):
        sampled = F.softmax(prediction, dim=2)
        # Select the symbol with the highest value ("probability")
        symbols = []

        for i in range(sampled.size(0)):
            if mask[i]:
                symbol = symbol_dict[torch.argmax(sampled[i, 0, :])]
                symbols.append(symbol["key"])
            else:
                symbols.append("")

        max_arity = 0
        for s in symbols:
            if s != "":
                symbol = next(d for d in symbol_dict if d["key"] == s)
                if symbol["arity"] > max_arity:
                    max_arity = symbol["arity"]

        child_mask = torch.empty([max_arity, mask.size(0)])
        for i in range(max_arity):
            ith_mask = torch.clone(mask)
            for j in range(ith_mask.size(0)):
                if symbols[j] != "":
                    symbol = next(d for d in symbol_dict if d["key"] == symbols[j])
                    if symbol["arity"] <= i:
                        ith_mask[j] = False
            child_mask[i] = ith_mask

        return symbols, child_mask


class GRU221(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRU221, self).__init__()
        self.wir = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.whr = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.wiz = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.whz = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.win = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.whn = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        torch.nn.init.xavier_uniform_(self.wir.weight)
        torch.nn.init.xavier_uniform_(self.whr.weight)
        torch.nn.init.xavier_uniform_(self.wiz.weight)
        torch.nn.init.xavier_uniform_(self.whz.weight)
        torch.nn.init.xavier_uniform_(self.win.weight)
        torch.nn.init.xavier_uniform_(self.whn.weight)

    def forward(self, x, h_sum, hs):
        #h = torch.cat(hs, dim=2)
        rs = []
        for h in hs:
            rs.append(torch.sigmoid(self.wir(x) + self.whr(h)))
        z = torch.sigmoid(self.wiz(x) + self.whz(h_sum))
        s = rs[0] * hs[0]
        first = True
        for r, h in zip(rs, hs):
            if not first:
                s += r * h
            first = False
        n = torch.tanh(self.win(x) + self.whn(s))
        return (1 - z) * n + z * h_sum


'''class GRU122(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRU122, self).__init__()
        self.hidden_size = hidden_size
        self.wir = nn.Linear(in_features=input_size, out_features=2*hidden_size)
        self.whr = nn.Linear(in_features=hidden_size, out_features=2*hidden_size)
        self.wiz = nn.Linear(in_features=input_size, out_features=2*hidden_size)
        self.whz = nn.Linear(in_features=hidden_size, out_features=2*hidden_size)
        self.win = nn.Linear(in_features=input_size, out_features=2*hidden_size)
        self.whn = nn.Linear(in_features=hidden_size, out_features=2*hidden_size)
        torch.nn.init.xavier_uniform_(self.wir.weight)
        torch.nn.init.xavier_uniform_(self.whr.weight)
        torch.nn.init.xavier_uniform_(self.wiz.weight)
        torch.nn.init.xavier_uniform_(self.whz.weight)
        torch.nn.init.xavier_uniform_(self.win.weight)
        torch.nn.init.xavier_uniform_(self.whn.weight)

    def forward(self, x, h):
        r = torch.sigmoid(self.wir(x) + self.whr(h))
        z = torch.sigmoid(self.wiz(x) + self.whz(h))
        n = torch.tanh(self.win(x) + r * self.whn(h))
        dh = h.repeat(1, 1, 2)
        out = (1 - z) * n + z * dh
        return torch.split(out, self.hidden_size, dim=2)'''

class GRUAncestral(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUAncestral, self).__init__()
        self.hidden_size = hidden_size
        self.wir = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.whr = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.wiz = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.whz = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.win = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.whn = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        torch.nn.init.xavier_uniform_(self.wir.weight)
        torch.nn.init.xavier_uniform_(self.whr.weight)
        torch.nn.init.xavier_uniform_(self.wiz.weight)
        torch.nn.init.xavier_uniform_(self.whz.weight)
        torch.nn.init.xavier_uniform_(self.win.weight)
        torch.nn.init.xavier_uniform_(self.whn.weight)

    def forward(self, x, h):
        r = torch.sigmoid(self.wir(x) + self.whr(h))
        z = torch.sigmoid(self.wiz(x) + self.whz(h))
        n = torch.tanh(self.win(x) + r * self.whn(h))
        out = (1 - z) * n + z * h
        return out

class GRUFraternal(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUFraternal, self).__init__()
        self.hidden_size = hidden_size
        self.wir = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.whr = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.wiz = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.whz = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.win = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.whn = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        torch.nn.init.xavier_uniform_(self.wir.weight)
        torch.nn.init.xavier_uniform_(self.whr.weight)
        torch.nn.init.xavier_uniform_(self.wiz.weight)
        torch.nn.init.xavier_uniform_(self.whz.weight)
        torch.nn.init.xavier_uniform_(self.win.weight)
        torch.nn.init.xavier_uniform_(self.whn.weight)

    def forward(self, x, h):
        r = torch.sigmoid(self.wir(x) + self.whr(h))
        z = torch.sigmoid(self.wiz(x) + self.whz(h))
        n = torch.tanh(self.win(x) + r * self.whn(h))
        out = (1 - z) * n + z * h
        return out
