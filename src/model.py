import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

from tree import Node, BatchedNode


class nHVAE(nn.Module):
    _symbols = None

    def __init__(self, input_size, output_size, max_arity, hidden_size=None):
        super(nHVAE, self).__init__()

        if hidden_size is None:
            hidden_size = output_size

        self.encoder = Encoder(input_size, hidden_size, output_size, max_arity)
        self.decoder = Decoder(output_size, hidden_size, input_size, max_arity)

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
        if nHVAE._symbols is None:
            raise Exception("To generate expression trees, a symbol library is needed. Please add it using the"
                            " HVAE.add_symbols method.")
        return self.decoder.decode(z, nHVAE._symbols)

    @staticmethod
    def add_symbols(symbols):
        nHVAE._symbols = symbols
        Node.add_symbols(symbols)
        BatchedNode.add_symbols(symbols)


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, max_arity):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.gru = GRU_N21(input_size=input_size, hidden_size=hidden_size)
        self.mu = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.logvar = nn.Linear(in_features=hidden_size, out_features=output_size)

        self.ws = []
        for i in range(max_arity):
            self.ws.append(nn.Linear(in_features=hidden_size, out_features=hidden_size))
            torch.nn.init.xavier_uniform_(self.ws[i].weight)

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
        children = [self.recursive_forward(t) for t in tree.children]
        if len(children) == 0:
            child_sum = torch.zeros(tree.target.size(0), 1, self.hidden_size)
            children.append(child_sum)
        else:
            child_sum = torch.zeros(children[0].size())
            for i in range(len(children)):
                child_sum = child_sum + self.ws[i](children[i])

        hidden = self.gru(tree.target, self.w_e(child_sum), children)
        hidden = hidden.mul(tree.mask[:, None, None])
        return hidden


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, max_arity):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_arity = max_arity
        self.h2o = nn.Linear(hidden_size, output_size)
        self.gru = GRU221(input_size=output_size, hidden_size=hidden_size)

        self.z2h = nn.Linear(input_size, hidden_size)
        self.w_d = nn.Linear(hidden_size, hidden_size)

        self.w_p = nn.Linear(hidden_size, 1)

        torch.nn.init.xavier_uniform_(self.h2o.weight)

        torch.nn.init.xavier_uniform_(self.z2h.weight)
        torch.nn.init.xavier_uniform_(self.w_d.weight)

        torch.nn.init.xavier_uniform_(self.w_p.weight)

    # Used during training to guide the learning process
    def forward(self, z, tree):
        hidden = self.z2h(z)
        s = torch.transpose(torch.stack([torch.zeros(hidden.shape[0])] * self.output_size), 0, 1)[:, None, :]
        self.recursive_forward(hidden, tree, s)
        return tree

    def recursive_forward(self, hidden_a, tree, s):
        p_f = torch.sigmoid(self.w_p(hidden_a))

        p_f_negation = [1 - i.item() for i in p_f[:, 0, 0]]
        pred_f = torch.cat((torch.Tensor(p_f_negation)[:, None, None], p_f), dim=2)

        prediction = self.h2o(hidden_a)
        symbol_probs = F.softmax(prediction, dim=2)

        tree.prediction = prediction
        tree.fraternal = {'prediction': pred_f, 'target': s[:, :, 0][:, :, None]}

        first = True
        for i in range(len(tree.children)):
            # computing new s values
            s = torch.zeros(hidden_a.shape[0]) if i+1 == len(tree.children) else torch.Tensor([0 if s == '' else 1 for s in tree.children[i + 1].symbols])
            s = torch.transpose(torch.stack([s] * self.output_size), 0, 1)[:, None, :]

            x = torch.zeros(symbol_probs.size()) if first else symbol_probs_f
            h = torch.zeros(hidden_a.size()) if first else hidden_f

            hidden = self.gru(symbol_probs, x, hidden_a, h)
            hidden_f, symbol_probs_f = self.recursive_forward(self.w_d(hidden), tree.children[i], s)
            first = False

        return hidden_a, symbol_probs

    # Used for inference to generate expression trees from latent vectors
    def decode(self, z, symbol_dict):
        with torch.no_grad():
            mask = torch.ones(z.size(0)).bool()
            hidden = self.z2h(z)
            symbols_multiarity = [s["key"] for s in symbol_dict if "min_arity" in s]
            batch, _, _, _ = self.recursive_decode(hidden, symbol_dict, mask, symbols_multiarity)
            return batch.to_expr_list()

    def recursive_decode(self, hidden_a, symbol_dict, mask, symbols_multiarity):
        p_f = torch.sigmoid(self.w_p(hidden_a))

        s = torch.Tensor([1 if i.item() >= 0.5 else 0 for i in p_f[:, 0, 0]])

        prediction = F.softmax(self.h2o(hidden_a), dim=2)
        # Sample symbol in a given node
        symbols, child_mask = self.sample_symbol(prediction, symbol_dict, mask)

        first = True
        children = []
        hasNextChild = True
        for i in range(child_mask.size(0)):
            if not hasNextChild: break
            if torch.any(child_mask[i]) and first:
                x, h = torch.zeros(prediction.size()), torch.zeros(hidden_a.size())
            elif not first:
                x, h = prediction_f, hidden_f
            hidden = self.gru(prediction, x, hidden_a, h)
            child, hidden_f, prediction_f, s_f = self.recursive_decode(self.w_d(hidden), symbol_dict, child_mask[i], symbols_multiarity)
            children.append(child)

            if i < len(child_mask) - 1 and not torch.any(child_mask[i+1]):
                hasNextChild = False
                for symbol in symbols_multiarity:
                    if symbol in symbols:
                        hasNextChild = True
                        break
                if hasNextChild:
                    hasNextChild = torch.sum(s_f).item() > 0
                    test = torch.Tensor([1 if sym in symbols_multiarity else 0 for sym in symbols])
                    child_mask[i+1] = s_f * test
            first = False

        node = BatchedNode()
        node.symbols = symbols
        node.children = children
        return node, hidden_a, prediction, s

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
                if "arity" in symbol and symbol["arity"] > max_arity:
                    max_arity = symbol["arity"]
                elif "arity" not in symbol:
                    max_arity = self.max_arity

        child_mask = torch.empty([max_arity, mask.size(0)])
        for i in range(max_arity):
            ith_mask = torch.clone(mask)
            for j in range(ith_mask.size(0)):
                if symbols[j] != "":
                    symbol = next(d for d in symbol_dict if d["key"] == symbols[j])
                    if "arity" in symbol and symbol["arity"] <= i:
                        ith_mask[j] = False
                    elif "arity" not in symbol and symbol["min_arity"] <= i:
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
        self.wxn = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.whn = nn.Linear(in_features=hidden_size, out_features=hidden_size)

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


class GRU221(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRU221, self).__init__()
        self.wxr = nn.Linear(in_features=2*input_size, out_features=hidden_size)
        self.whr = nn.Linear(in_features=2*hidden_size, out_features=hidden_size)
        self.wxz = nn.Linear(in_features=2*input_size, out_features=hidden_size)
        self.whz = nn.Linear(in_features=2*hidden_size, out_features=hidden_size)
        self.wxn = nn.Linear(in_features=2*input_size, out_features=hidden_size)
        self.whn = nn.Linear(in_features=2*hidden_size, out_features=hidden_size)
        torch.nn.init.xavier_uniform_(self.wxr.weight)
        torch.nn.init.xavier_uniform_(self.whr.weight)
        torch.nn.init.xavier_uniform_(self.wxz.weight)
        torch.nn.init.xavier_uniform_(self.whz.weight)
        torch.nn.init.xavier_uniform_(self.wxn.weight)
        torch.nn.init.xavier_uniform_(self.whn.weight)

    def forward(self, x1, x2, h1, h2):
        h = torch.cat([h1, h2], dim=2)
        x = torch.cat([x1, x2], dim=2)
        r = torch.sigmoid(self.wxr(x) + self.whr(h))
        z = torch.sigmoid(self.wxz(x) + self.whz(h))
        n = torch.tanh(self.wxn(x) + r * self.whn(h))
        return (1 - z) * n + (z / 2) * h1 + (z / 2) * h2

