import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

from tree import Node, BatchedNode


class HVAE(nn.Module):
    _symbols = None

    def __init__(self, input_size, output_size, max_arity, hidden_size=None):
        super(HVAE, self).__init__()

        if hidden_size is None:
            hidden_size = output_size

        self.encoder = Encoder(input_size, hidden_size, output_size, max_arity)
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
    def __init__(self, input_size, hidden_size, output_size, max_arity):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.gru = GRU_N21(input_size=input_size, hidden_size=hidden_size)
        self.mu = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.logvar = nn.Linear(in_features=hidden_size, out_features=output_size)

        self.w_ks = []
        for i in range(max_arity):
            self.w_ks.append(nn.Linear(in_features=hidden_size, out_features=hidden_size))
            torch.nn.init.xavier_uniform_(self.w_ks[i].weight)

        self.w_e = nn.Linear(hidden_size, hidden_size)

        torch.nn.init.xavier_uniform_(self.w_e.weight)

        torch.nn.init.xavier_uniform_(self.mu.weight)
        torch.nn.init.xavier_uniform_(self.logvar.weight)

    def forward(self, tree):
        if tree.target is None:
            tree.add_target_vectors()

        tree_encoding = self.recursive_forward(tree)
        mu = self.mu(tree_encoding)
        logvar = self.logvar(tree_encoding)
        return mu, logvar

    def recursive_forward(self, tree):
        children = []
        for t in tree.children:
            children.append(self.recursive_forward(t))
        if len(children) == 0:
            children.append(torch.zeros(tree.target.size(0), 1, self.hidden_size))
            child_sum = children[0]
        else:
            child_sum = torch.zeros(children[0].size())
            i = 0
            for c in children:
                child_sum = child_sum + self.w_ks[i](c)
                i += 1

        hidden = self.gru(tree.target, self.w_e(child_sum), children)
        hidden = hidden.mul(tree.mask[:, None, None])
        return hidden


# nHVAE (simple) decoder
'''class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.h2o = nn.Linear(hidden_size, output_size)
        self.gru_a = GRUAncestral(input_size=output_size, hidden_size=hidden_size)
        self.gru_s = GRUFraternal(input_size=output_size, hidden_size=hidden_size)

        self.z2h = nn.Linear(input_size, hidden_size)
        self.w_d = nn.Linear(hidden_size, hidden_size)

        torch.nn.init.xavier_uniform_(self.h2o.weight)
        torch.nn.init.xavier_uniform_(self.z2h.weight)
        torch.nn.init.xavier_uniform_(self.w_d.weight)

    # Used during training to guide the learning process
    def forward(self, z, tree):
        hidden = self.z2h(z)
        self.recursive_forward(hidden, tree)
        return tree

    def recursive_forward(self, hidden_a, tree):
        prediction = self.h2o(hidden_a)
        symbol_probs = F.softmax(prediction, dim=2)

        tree.prediction = prediction

        first = True
        for t in tree.children:
            if t is not None:
                if first:
                    hidden_f, symbol_probs_f = self.recursive_forward(self.w_d(hidden_a), t)
                else:
                    hidden_f = self.gru_s(symbol_probs_f, hidden_f)
                    hidden = hidden_f + self.gru_a(symbol_probs, hidden_a)
                    hidden_f, symbol_probs_f = self.recursive_forward(self.w_d(hidden), t)
                first = False

        return hidden_a, symbol_probs

    # Used for inference to generate expression trees from latent vectors
    def decode(self, z, symbol_dict):
        with torch.no_grad():
            mask = torch.ones(z.size(0)).bool()
            hidden = self.z2h(z)
            batch, _, _ = self.recursive_decode(hidden, symbol_dict, mask)
            return batch.to_expr_list()

    def recursive_decode(self, hidden_a, symbol_dict, mask):
        prediction = F.softmax(self.h2o(hidden_a), dim=2)
        # Sample symbol in a given node
        symbols, child_mask = self.sample_symbol(prediction, symbol_dict, mask)

        first = True
        children = []
        for i in range(child_mask.size(0)):
            if first:
                if torch.any(child_mask[i]):
                    child, hidden_f, prediction_f = self.recursive_decode(self.w_d(hidden_a), symbol_dict, child_mask[i])
                    children.append(child)
            elif torch.any(child_mask[i]):
                hidden_f = self.gru_s(prediction_f, hidden_f)
                hidden = hidden_f + self.gru_a(prediction, hidden_a)
                child, hidden_f, prediction_f = self.recursive_decode(self.w_d(hidden), symbol_dict, child_mask[i])
                children.append(child)
            first = False

        node = BatchedNode()
        node.symbols = symbols
        node.children = children
        return node, hidden_a, prediction

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

        return symbols, child_mask'''


# nHVAE (complex) decoder
class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.h2o = nn.Linear(hidden_size, output_size)
        self.gru_a = GRUAncestral(input_size=output_size, hidden_size=hidden_size)
        self.gru_s = GRUFraternal(input_size=output_size, hidden_size=hidden_size)

        self.w_c = nn.Linear(input_size, hidden_size)
        self.w_d = nn.Linear(hidden_size, hidden_size)

        self.z2h1 = nn.Linear(input_size, hidden_size)
        self.z2h2 = nn.Linear(input_size, hidden_size)

        torch.nn.init.xavier_uniform_(self.h2o.weight)
        #torch.nn.init.xavier_uniform_(self.z2h.weight)

        torch.nn.init.xavier_uniform_(self.w_c.weight)
        torch.nn.init.xavier_uniform_(self.w_d.weight)
        torch.nn.init.xavier_uniform_(self.z2h1.weight)
        torch.nn.init.xavier_uniform_(self.z2h2.weight)

    # Used during training to guide the learning process
    def forward(self, z, tree):
        hidden = z
        self.recursive_forward(self.z2h1(hidden), self.z2h2(hidden), tree)
        return tree

    # hidden_a represents the additional hidden code c
    def recursive_forward(self, hidden_a, hidden, tree):
        prediction = self.h2o(hidden)
        symbol_probs = F.softmax(prediction, dim=2)

        tree.prediction = prediction

        hidden_final = hidden

        first = True
        for t in tree.children:
            if t is not None:
                if first:
                    hidden_f, symbol_probs_f = self.recursive_forward(self.w_c(hidden_a), self.w_d(hidden), t)
                else:
                    hidden_f = self.gru_s(symbol_probs_f, hidden_f)
                    hidden = hidden_f + self.gru_a(symbol_probs, hidden_a)
                    hidden_f, symbol_probs_f = self.recursive_forward(self.w_c(hidden_a), self.w_d(hidden), t)
                first = False

        return hidden_final, symbol_probs

    # Used for inference to generate expression trees from latent vectors
    def decode(self, z, symbol_dict):
        with torch.no_grad():
            mask = torch.ones(z.size(0)).bool()
            hidden = z
            batch, _, _ = self.recursive_decode(self.z2h1(hidden), self.z2h2(hidden), symbol_dict, mask)
            return batch.to_expr_list()

    def recursive_decode(self, hidden_a, hidden, symbol_dict, mask):
        prediction = F.softmax(self.h2o(hidden), dim=2)
        # Sample symbol in a given node
        symbols, child_mask = self.sample_symbol(prediction, symbol_dict, mask)

        hidden_final = hidden

        first = True
        children = []
        for i in range(child_mask.size(0)):
            if first:
                if torch.any(child_mask[i]):
                    child, hidden_f, prediction_f = self.recursive_decode(self.w_c(hidden_a), self.w_d(hidden), symbol_dict,
                                                                child_mask[i])
                    children.append(child)
            elif torch.any(child_mask[i]):
                hidden_f = self.gru_s(prediction_f, hidden_f)
                hidden = hidden_f + self.gru_a(prediction, hidden_a)
                child, hidden_f, prediction_f = self.recursive_decode(self.w_c(hidden_a), self.w_d(hidden), symbol_dict,
                                                            child_mask[i])
                children.append(child)
            first = False

        node = BatchedNode()
        node.symbols = symbols
        node.children = children
        return node, hidden_final, prediction

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


class GRU_N21(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRU_N21, self).__init__()
        self.wxr = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.whr = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.wxz = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.whz = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.wxn = nn.Linear(in_features=input_size, out_features=hidden_size, bias=False)
        self.whn = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)

        torch.nn.init.xavier_uniform_(self.wxr.weight)
        torch.nn.init.xavier_uniform_(self.whr.weight)
        torch.nn.init.xavier_uniform_(self.wxz.weight)
        torch.nn.init.xavier_uniform_(self.whz.weight)
        torch.nn.init.xavier_uniform_(self.wxn.weight)
        torch.nn.init.xavier_uniform_(self.whn.weight)

    def forward(self, x, h_sum, hs):
        rs = []
        for h in hs:
            rs.append(torch.sigmoid(self.wxr(x) + self.whr(h)))
        z = torch.sigmoid(self.wxz(x) + self.whz(h_sum))
        s = torch.zeros(hs[0].size())
        for r, h in zip(rs, hs):
            s += r * h
        n = torch.tanh(self.wxn(x) + self.whn(s))
        return (1 - z) * n + z * h_sum


class GRUAncestral(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUAncestral, self).__init__()
        self.hidden_size = hidden_size
        self.wxr = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.whr = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.wxz = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.whz = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.wxn = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.whn = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        torch.nn.init.xavier_uniform_(self.wxr.weight)
        torch.nn.init.xavier_uniform_(self.whr.weight)
        torch.nn.init.xavier_uniform_(self.wxz.weight)
        torch.nn.init.xavier_uniform_(self.whz.weight)
        torch.nn.init.xavier_uniform_(self.wxn.weight)
        torch.nn.init.xavier_uniform_(self.whn.weight)

    def forward(self, x, h):
        r = torch.sigmoid(self.wxr(x) + self.whr(h))
        z = torch.sigmoid(self.wxz(x) + self.whz(h))
        n = torch.tanh(self.wxn(x) + r * self.whn(h))
        out = (1 - z) * n + z * h
        return out


class GRUFraternal(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUFraternal, self).__init__()
        self.hidden_size = hidden_size
        self.wxr = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.whr = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.wxz = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.whz = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.wxn = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.whn = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        torch.nn.init.xavier_uniform_(self.wxr.weight)
        torch.nn.init.xavier_uniform_(self.whr.weight)
        torch.nn.init.xavier_uniform_(self.wxz.weight)
        torch.nn.init.xavier_uniform_(self.whz.weight)
        torch.nn.init.xavier_uniform_(self.wxn.weight)
        torch.nn.init.xavier_uniform_(self.whn.weight)

    def forward(self, x, h):
        r = torch.sigmoid(self.wxr(x) + self.whr(h))
        z = torch.sigmoid(self.wxz(x) + self.whz(h))
        n = torch.tanh(self.wxn(x) + r * self.whn(h))
        out = (1 - z) * n + z * h
        return out

