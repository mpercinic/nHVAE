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

        self.wms = []
        for i in range(max_arity):
            self.wms.append(nn.Linear(in_features=hidden_size, out_features=hidden_size))
            torch.nn.init.xavier_uniform_(self.wms[i].weight)
        self.w = nn.Linear(in_features=hidden_size, out_features=1)
        self.w_a = nn.Linear(in_features=hidden_size, out_features=hidden_size)

        torch.nn.init.xavier_uniform_(self.w.weight)
        torch.nn.init.xavier_uniform_(self.w_a.weight)

        #self.l = nn.Linear(hidden_size, hidden_size)
        #torch.nn.init.xavier_uniform_(self.l.weight)

        '''self.ms = []
        for i in range(max_arity):
            self.ms.append(nn.Linear(in_features=hidden_size, out_features=hidden_size))
            torch.nn.init.xavier_uniform_(self.ms[i].weight)'''

        '''self.ms = []
        for i in range(max_arity):
            m_i = []
            for j in range(i + 1):
                m_i.append(nn.Linear(in_features=hidden_size, out_features=hidden_size))
                torch.nn.init.xavier_uniform_(m_i[j].weight)
            self.ms.append(m_i)'''

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
            ms = []
            i = 0
            for c in children:
                ms.append(torch.tanh(self.wms[i](c)))
                i += 1
            ms2 = []
            for m in ms:
                ms2.append(self.w(m))
            aks = []
            ms_sum = sum(ms2)
            for m in ms2:
                aks.append(m / ms_sum)
            g = torch.zeros(children[0].size())
            for a, c in zip(aks, children):
                g += a * c
            child_sum = torch.tanh(self.w_a(g))
            '''child_sum = torch.zeros(children[0].size())
            i = 0
            for c in children:
                child_sum = child_sum + self.ms[i](c) * c
                i += 1'''
            '''child_sum = torch.zeros(children[0].size())
            i = 0
            for c in children:
                child_sum = child_sum + self.ms[len(children) - 1][i](c) * c
                i += 1'''

        # hidden = self.gru(tree.target, self.l(child_sum), children)
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
        self.u_a = nn.Linear(hidden_size, hidden_size, bias=False)
        self.u_f = nn.Linear(hidden_size, hidden_size, bias=False)
        #self.gru = GRU221(input_size=output_size, hidden_size=hidden_size)

        self.f_init = nn.Linear(hidden_size, hidden_size)
        self.a = nn.Linear(hidden_size, hidden_size)
        self.h = nn.Linear(hidden_size, hidden_size)

        #self.test = nn.Linear(output_size, output_size)
        #torch.nn.init.xavier_uniform_(self.test.weight)

        torch.nn.init.xavier_uniform_(self.z2h.weight)
        torch.nn.init.xavier_uniform_(self.h2o.weight)
        torch.nn.init.xavier_uniform_(self.u_a.weight)
        torch.nn.init.xavier_uniform_(self.u_f.weight)

        torch.nn.init.xavier_uniform_(self.f_init.weight)
        torch.nn.init.xavier_uniform_(self.a.weight)
        torch.nn.init.xavier_uniform_(self.h.weight)

    # Used during training to guide the learning process
    def forward(self, z, tree):
        hidden = self.z2h(z)
        self.recursive_forward(self.a(hidden), self.h(hidden), tree)
        return tree

    def recursive_forward(self, hidden_a, hidden, tree):
        prediction = self.h2o(hidden)
        symbol_probs = F.softmax(prediction, dim=2)

        tree.prediction = prediction

        #symbol_probs_a = F.softmax(self.h2o(hidden_a), dim=2)
        hidden_a_i = self.gru_ancestral(symbol_probs, hidden_a)
        first = True
        for t in tree.children:
            if t is not None:
                if first:
                    symbol_probs_f = self.recursive_forward(self.a(hidden_a_i), self.h(hidden_a_i), t)
                    hidden_f = self.f_init(hidden_a_i)
                else:
                    hidden_f = self.gru_fraternal(symbol_probs_f, hidden_f)
                    hidden = torch.tanh(self.u_f(hidden_f) + self.u_a(hidden_a_i))
                    #hidden = self.gru(symbol_probs, hidden_a_i, hidden_f)
                    symbol_probs_f = self.recursive_forward(self.a(hidden_a_i), self.h(hidden), t)
                first = False

        return symbol_probs

    # Used for inference to generate expression trees from latent vectorS
    def decode(self, z, symbol_dict):
        with torch.no_grad():
            mask = torch.ones(z.size(0)).bool()
            hidden = self.z2h(z)
            batch, _ = self.recursive_decode(self.a(hidden), self.h(hidden), symbol_dict, mask)
            return batch.to_expr_list()

    def recursive_decode(self, hidden_a, hidden, symbol_dict, mask):
        prediction = F.softmax(self.h2o(hidden), dim=2)
        # Sample symbol in a given node
        symbols, child_mask = self.sample_symbol(prediction, symbol_dict, mask)

        first = True
        #prediction_a = F.softmax(self.h2o(hidden_a), dim=2)
        hidden_a_i = self.gru_ancestral(prediction, hidden_a)
        children = []
        for i in range(child_mask.size(0)):
            if first:
                hidden_f = self.f_init(hidden_a_i)
                if torch.any(child_mask[i]):
                    child, prediction_f = self.recursive_decode(self.a(hidden_a_i), self.h(hidden_a_i), symbol_dict, child_mask[i])
                    children.append(child)
            elif torch.any(child_mask[i]):
                hidden_f = self.gru_fraternal(prediction_f, hidden_f)
                hidden = torch.tanh(self.u_f(hidden_f) + self.u_a(hidden_a_i))
                #hidden = self.gru(prediction, hidden_a_i, hidden_f)
                child, prediction_f = self.recursive_decode(self.a(hidden_a_i), self.h(hidden), symbol_dict, child_mask[i])
                children.append(child)
            first = False

        node = BatchedNode()
        node.symbols = symbols
        node.children = children
        return node, prediction

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
        rs = []
        for h in hs:
            rs.append(torch.sigmoid(self.wir(x) + self.whr(h)))
        z = torch.sigmoid(self.wiz(x) + self.whz(h_sum))
        s = torch.zeros(hs[0].size())
        for r, h in zip(rs, hs):
            s += r * h
        n = torch.tanh(self.win(x) + self.whn(s))
        return (1 - z) * n + z * h_sum


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


class GRU221(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRU221, self).__init__()
        self.wir = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.whr = nn.Linear(in_features=2 * hidden_size, out_features=hidden_size)
        self.wiz = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.whz = nn.Linear(in_features=2 * hidden_size, out_features=hidden_size)
        self.win = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.whn = nn.Linear(in_features=2 * hidden_size, out_features=hidden_size)
        torch.nn.init.xavier_uniform_(self.wir.weight)
        torch.nn.init.xavier_uniform_(self.whr.weight)
        torch.nn.init.xavier_uniform_(self.wiz.weight)
        torch.nn.init.xavier_uniform_(self.whz.weight)
        torch.nn.init.xavier_uniform_(self.win.weight)
        torch.nn.init.xavier_uniform_(self.whn.weight)

        self.w1 = nn.Linear(in_features=hidden_size, out_features=1)
        self.w2 = nn.Linear(in_features=hidden_size, out_features=1)
        torch.nn.init.xavier_uniform_(self.w1.weight)
        torch.nn.init.xavier_uniform_(self.w2.weight)

    def forward(self, x, h1, h2):
        h = torch.cat([h1, h2], dim=2)
        r = torch.sigmoid(self.wir(x) + self.whr(h))
        z = torch.sigmoid(self.wiz(x) + self.whz(h))
        n = torch.tanh(self.win(x) + r * self.whn(h))

        a1 = self.w1(h1)
        a2 = self.w2(h2)
        h_sum = a1 * h1 + a2 * h2

        return (1 - z) * n + h_sum
