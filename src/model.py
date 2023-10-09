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
        # lengths = []
        for t in tree.children:
            children.append(self.recursive_forward(t))
            # lengths.append(t.batched_len())
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


        '''left = self.recursive_forward(tree.left) if tree.left is not None \
            else torch.zeros(tree.target.size(0), 1, self.hidden_size)
        right = self.recursive_forward(tree.right) if tree.right is not None \
            else torch.zeros(tree.target.size(0), 1, self.hidden_size)'''

        # left = left.mul(tree.mask[:, None, None])
        # right = right.mul(tree.mask[:, None, None])

        #hidden = self.gru(tree.target, left, right)

        hidden = self.gru(tree.target, child_sum, children)
        hidden = hidden.mul(tree.mask[:, None, None])
        return hidden


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.z2h = nn.Linear(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        # self.gru = GRU122(input_size=output_size, hidden_size=hidden_size)
        self.gru_ancestral = GRUAncestral(input_size=output_size, hidden_size=hidden_size)
        self.gru_fraternal = GRUFraternal(input_size=output_size, hidden_size=hidden_size)
        self.u_a = nn.Linear(hidden_size, hidden_size)
        self.u_f = nn.Linear(hidden_size, hidden_size)
        self.u_pred = nn.Linear(hidden_size, 1)
        # self.sibling_threshold = 0.5

        torch.nn.init.xavier_uniform_(self.z2h.weight)
        torch.nn.init.xavier_uniform_(self.h2o.weight)
        torch.nn.init.xavier_uniform_(self.u_a.weight)
        torch.nn.init.xavier_uniform_(self.u_f.weight)
        torch.nn.init.xavier_uniform_(self.u_pred.weight)

    # Used during training to guide the learning process
    def forward(self, z, tree):
        # start = time.time()
        hidden = self.z2h(z)
        prediction = self.h2o(hidden)
        init_prediction = torch.zeros(prediction.size())
        tree.initialize_predictions(init_prediction)
        self.recursive_forward(hidden, tree)
        # end = time.time()
        # print("Decoder: " + str(end - start))
        return tree

    def recursive_forward(self, hidden_a, tree):
        prediction = self.h2o(hidden_a)
        symbol_probs = F.softmax(prediction, dim=2)
        tree.prediction = prediction

        hidden_a_i = self.gru_ancestral(symbol_probs, hidden_a)
        first = True
        for t in tree.children:
            if t is not None:
                if first:
                    symbol_probs_f = self.recursive_forward(hidden_a_i, t)
                    sibling_prediction = torch.sigmoid(self.u_pred(hidden_a_i))
                    if random() > torch.max(sibling_prediction):
                        break
                    hidden_f = hidden_a_i
                else:
                    hidden_f = self.gru_fraternal(symbol_probs_f, hidden_f)
                    hidden = torch.tanh(self.u_f(hidden_f) + self.u_a(hidden_a))
                    sibling_prediction = torch.sigmoid(self.u_pred(hidden))
                    if random() > torch.max(sibling_prediction):
                        break
                    symbol_probs_f = self.recursive_forward(hidden, t)
                first = False
        return symbol_probs

    # Used for inference to generate expression trees from latent vectorS
    def decode(self, z, symbol_dict):
        with torch.no_grad():
            mask = torch.ones(z.size(0)).bool()
            num_children_mask = torch.ones(z.size(0))
            hidden = self.z2h(z)
            batch = self.recursive_decode(hidden, symbol_dict, mask, num_children_mask)
            return batch.to_expr_list()

    def recursive_decode(self, hidden_a, symbol_dict, mask, num_children_mask):
        prediction = F.softmax(self.h2o(hidden_a), dim=2)
        # Sample symbol in a given node
        # print(num_children_mask)
        symbols, child_mask, num_children_mask_new = self.sample_symbol(prediction, symbol_dict, mask)

        first = True
        hidden_a_i = self.gru_ancestral(prediction, hidden_a)
        has_right_sibling = True
        children = []
        while has_right_sibling:
            if first:
                hidden_f = hidden_a_i
                sibling_prediction = torch.sigmoid(self.u_pred(hidden_a_i))
                '''for i in range(len(sibling_prediction)):
                    right_sibling_bool = 1 if mask[i] and random() <= sibling_prediction[i][0][0] else 0
                    right_sibling_mask.append(right_sibling_bool)'''
                # has_right_sibling = True if sum(right_sibling_mask) > 0 else False
                has_right_sibling = True if torch.any(mask) and random() <= torch.max(sibling_prediction) else False
                if has_right_sibling:
                    right_sibling_mask = []
                    for i in range(len(sibling_prediction)):
                        right_sibling_bool = 1 if mask[i] and random() <= sibling_prediction[i][0][0] else 0
                        right_sibling_mask.append(right_sibling_bool)
                    for i in range(len(right_sibling_mask)):
                        mask[i] = True if right_sibling_mask[i] == 1 else False
                # print(child_mask)
                if torch.any(child_mask):
                    child = self.recursive_decode(hidden_a_i, symbol_dict, child_mask, num_children_mask_new)
                    children.append(child)
            else:
                prediction_f = F.softmax(self.h2o(hidden_f), dim=2)
                hidden_f = self.gru_fraternal(prediction_f, hidden_f)
                hidden = torch.tanh(self.u_f(hidden_f) + self.u_a(hidden_a))
                prediction_2 = F.softmax(self.h2o(hidden), dim=2)
                _, child_mask, num_children_mask_new = self.sample_symbol(prediction_2, symbol_dict, mask)
                # print("sibling: " + str(sibling_prediction[0]))
                # right_sibling_mask = []
                sibling_prediction = torch.sigmoid(self.u_pred(hidden))
                has_right_sibling = True if torch.any(mask) and random() <= torch.max(sibling_prediction) else False
                '''for i in range(len(sibling_prediction)):
                    right_sibling_bool = 1 if mask[i] and random() <= sibling_prediction[i][0][0] else 0
                    right_sibling_mask.append(right_sibling_bool)'''
                # has_right_sibling = True if sum(right_sibling_mask) > 0 else False
                if has_right_sibling:
                    right_sibling_mask = []
                    for i in range(len(sibling_prediction)):
                        right_sibling_bool = 1 if mask[i] and random() <= sibling_prediction[i][0][0] else 0
                        right_sibling_mask.append(right_sibling_bool)
                    for i in range(len(right_sibling_mask)):
                        mask[i] = True if right_sibling_mask[i] == 1 else False
                    child = self.recursive_decode(hidden, symbol_dict, child_mask, num_children_mask_new)
                    children.append(child)
                # print(child_mask)
                '''if torch.any(child_mask):
                    child = self.recursive_decode(hidden, symbol_dict, child_mask, num_children_mask_new)
                    children.append(child)'''
            first = False

        node = BatchedNode()
        node.symbols = symbols
        node.children = children
        return node

    '''first2 = True
    for i in range(len(num_children_mask)):
        if num_children_mask[i] == 2:
            right_sibling_bool = 1
            num_children_mask[i] -= 1
        elif num_children_mask[i] == 11 or num_children_mask[i] == -1:
            right_sibling_bool = 0
            num_children_mask[i] = -1  # perhaps not needed
        else:
            if first2:
                sibling_prediction = torch.sigmoid(self.u_pred(hidden_a_i))
                first2 = False
            print(sibling_prediction[0][0])
            right_sibling_bool = 1 if mask[i] and random() <= sibling_prediction[i][0][0] else 0
        if num_children_mask[i] == 1:
            num_children_mask[i] -= -1
        right_sibling_mask.append(right_sibling_bool)'''

    def sample_symbol(self, prediction, symbol_dict, mask):
        sampled = F.softmax(prediction, dim=2)
        # Select the symbol with the highest value ("probability")
        symbols = []
        child_mask = torch.clone(mask)
        num_children_mask = torch.zeros(child_mask.size(0))

        for i in range(sampled.size(0)):
            if mask[i]:
                symbol = symbol_dict[torch.argmax(sampled[i, 0, :])]
                symbols.append(symbol["symbol"])
                if symbol["type"].value is SymType.Var.value or symbol["type"].value is SymType.Const.value:
                    child_mask[i] = False
                    num_children_mask[i] = -1
                elif symbol["type"].value is SymType.Fun.value:
                    num_children_mask[i] = 11
                elif symbol["type"].value is SymType.Operator.value and symbol != "^":
                    num_children_mask[i] = 2
                elif symbol == "^":
                    num_children_mask[i] = 1
            else:
                symbols.append("")
        # num_children_mask values and meaning:
        # -1: no children
        # 0: 0 or more children
        # 1: 1 or more children
        # 11: only 1 child
        # 2: 2 or more children

        print(symbols)

        return symbols, child_mask, num_children_mask


class GRU221(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRU221, self).__init__()
        '''whrs, whzs, whns = [], [], []
        for i in range(n):
            whrs.append(nn.Linear(in_features=hidden_size, out_features=hidden_size))
            whzs.append(nn.Linear(in_features=hidden_size, out_features=hidden_size))
            whns.append(nn.Linear(in_features=hidden_size, out_features=hidden_size))
        self.whrs = whrs
        self.whzs = whzs
        self.whns = whns'''
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
        '''for i in range(n):
            torch.nn.init.xavier_uniform_(self.whrs[i].weight)
            torch.nn.init.xavier_uniform_(self.whzs[i].weight)
            torch.nn.init.xavier_uniform_(self.whns[i].weight)'''

    '''def forward(self, x, h1, h2):
        h = torch.cat([h1, h2], dim=2)
        r = torch.sigmoid(self.wir(x) + self.whr(h))
        z = torch.sigmoid(self.wiz(x) + self.whz(h))
        n = torch.tanh(self.win(x) + r * self.whn(h))
        return (1 - z) * n + (z / 2) * h1 + (z / 2) * h2'''

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
