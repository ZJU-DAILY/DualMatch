import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter


class Embedding_init(nn.Module):
    @staticmethod
    def init_emb(row, col):
        w = torch.empty(row, col)
        torch.nn.init.normal_(w)
        w = torch.nn.functional.normalize(w)
        entities_emb = nn.Parameter(w)
        return entities_emb


class OverAll(nn.Module):
    def __init__(self, node_size, node_hidden,
                 rel_size, rel_hidden,
                 time_size,
                 triple_size,
                 rel_matrix,
                 ent_matrix,
                 time_matrix,
                 dropout_rate=0, depth=2, dropout_time=0.5,
                 device='cpu'
                 ):
        super(OverAll, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout_time = dropout_time

        # new adding
        # rel_or_time in GraphAttention.forward

        self.e_encoder = GraphAttention(node_size, rel_size, triple_size, time_size, depth=depth, device=device,
                                        dim=node_hidden)
        self.r_encoder = GraphAttention(node_size, rel_size, triple_size, time_size, depth=depth, device=device,
                                        dim=node_hidden)
        # self.t_encoder = GraphAttention(node_size, rel_size, triple_size, time_size, depth=depth, device=device,
        #                                 dim=node_hidden)

        self.ent_adj = self.get_spares_matrix_by_index(ent_matrix, (node_size, node_size))
        self.rel_adj = self.get_spares_matrix_by_index(rel_matrix, (node_size, rel_size))
        self.time_adj = self.get_spares_matrix_by_index(time_matrix, (node_size, time_size))

        self.ent_emb = self.init_emb(node_size, node_hidden)
        self.rel_emb = self.init_emb(rel_size, node_hidden)
        self.time_emb = self.init_emb(time_size, node_hidden)
        self.try_emb = self.init_emb(1, node_hidden)
        self.device = device
        self.ent_adj, self.rel_adj, self.time_adj = \
            map(lambda x: x.to(device), [self.ent_adj, self.rel_adj, self.time_adj])

    # get prepared
    @staticmethod
    def get_spares_matrix_by_index(index, size):
        index = torch.LongTensor(index)
        adj = torch.sparse.FloatTensor(torch.transpose(index, 0, 1),
                                       torch.ones_like(index[:, 0], dtype=torch.float), size)
        # dim ??
        return torch.sparse.softmax(adj, dim=1)

    @staticmethod
    def init_emb(*size):
        entities_emb = nn.Parameter(torch.randn(size))
        torch.nn.init.xavier_normal_(entities_emb)
        return entities_emb

    def forward(self, inputs):
        # inputs = [adj_matrix, r_index, r_val, rel_matrix, ent_matrix, train_pairs]
        ent_feature = torch.matmul(self.ent_adj, self.ent_emb)
        rel_feature = torch.matmul(self.rel_adj, self.rel_emb)
        time_feature = torch.matmul(self.time_adj, self.time_emb)
        # ent_feature = torch.cat([ent_feature, rel_feature, time_feature], dim=1)

        # note that time_feature and rel_feature is has the same shape of ent_feature
        # the dim = node_hidden, the shape[0] = # of entities
        # They are obtained by gather the linked rel/time of an entity

        adj_input = inputs[0]
        r_index = inputs[1]
        r_val = inputs[2]
        t_index = inputs[3]

        opt = [self.rel_emb, adj_input, r_index, r_val]
        opt2 = [self.time_emb, adj_input, t_index, r_val]
        # attention opt_1 or 2
        out_feature_ent = self.e_encoder([ent_feature] + opt)
        out_feature_rel = self.r_encoder([rel_feature] + opt)
        out_feature_time = self.e_encoder([time_feature] + opt2, 1)
        # out_feature_ent2 = self.e_encoder([ent_feature] + opt)
        # out_feature_rel2 = self.r_encoder([rel_feature] + opt)
        # out_feature_time2 = self.t_encoder([time_feature] + opt2, 1)
        # out_feature_time = self.e_encoder([time_feature] + opt)
        # out_feature_time = F.dropout(out_feature_time, p=self.dropout_time, training=self.training)

        from config import global_args
        if global_args.dual_no_time:
            out_feature_overall = out_feature_rel
        else:
            out_feature_overall = (out_feature_rel + out_feature_time) / 2
        out_feature = torch.cat((out_feature_ent, out_feature_overall), dim=-1)
        out_feature = F.dropout(out_feature, p=self.dropout_rate, training=self.training)
        return out_feature


class GraphAttention(nn.Module):
    def __init__(self, node_size, rel_size, triple_size, time_size,
                 activation=torch.tanh, use_bias=True,
                 attn_heads=1, dim=100,
                 depth=1, device='cpu'):
        super(GraphAttention, self).__init__()
        self.node_size = node_size
        self.activation = activation
        self.rel_size = rel_size

        self.time_size = time_size

        self.triple_size = triple_size
        self.use_bias = use_bias
        self.attn_heads = attn_heads
        self.attn_heads_reduction = 'concat'
        self.depth = depth
        self.device = device
        self.attn_kernels = []

        node_F = dim
        rel_F = dim
        self.ent_F = node_F
        ent_F = self.ent_F

        # gate kernel Eq 9 M
        self.gate_kernel = OverAll.init_emb(ent_F * (self.depth + 1), ent_F * (self.depth + 1))
        self.proxy = OverAll.init_emb(64, node_F * (self.depth + 1))
        if self.use_bias:
            self.bias = OverAll.init_emb(1, ent_F * (self.depth + 1))
        for d in range(self.depth):
            self.attn_kernels.append([])
            for h in range(self.attn_heads):
                attn_kernel = OverAll.init_emb(node_F, 1)
                self.attn_kernels[d].append(attn_kernel.to(device))

    def forward(self, inputs, rel_or_time=0):
        outputs = []
        features = inputs[0]
        rel_emb = inputs[1]
        adj_index = inputs[2]  # adj
        index = torch.tensor(adj_index, dtype=torch.int64)
        index = index.to(self.device)
        # adj = torch.sparse.FloatTensor(torch.LongTensor(index),
        #                                torch.FloatTensor(torch.ones_like(index[:,0])),
        #                                (self.node_size, self.node_size))
        sparse_indices = inputs[3]  # relation index  i.e. r_index
        sparse_val = inputs[4]  # relation value  i.e. r_val

        features = self.activation(features)
        outputs.append(features)

        for l in range(self.depth):
            features_list = []
            for head in range(self.attn_heads):
                attention_kernel = self.attn_kernels[l][head]
                ####
                col = self.rel_size if rel_or_time == 0 else self.time_size
                rels_sum = torch.sparse.FloatTensor(
                    torch.transpose(torch.LongTensor(sparse_indices), 0, 1),
                    torch.FloatTensor(sparse_val),
                    (self.triple_size, col)
                )  # relation matrix
                rels_sum = rels_sum.to(self.device)
                rels_sum = torch.matmul(rels_sum, rel_emb)
                neighs = features[index[:, 1]]
                # selfs = features[index[:, 0]]
                rels_sum = F.normalize(rels_sum, p=2, dim=1)
                neighs = neighs - 2 * torch.sum(neighs * rels_sum, 1, keepdim=True) * rels_sum

                # Eq.3
                att1 = torch.squeeze(torch.matmul(rels_sum, attention_kernel), dim=-1)
                att = torch.sparse.FloatTensor(torch.transpose(index, 0, 1), att1, (self.node_size, self.node_size))
                # ??? dim ??
                att = torch.sparse.softmax(att, dim=1)
                # ?
                # print(att1)
                # print(att.data)
                new_features = torch_scatter.scatter_add(
                    torch.transpose(neighs * torch.unsqueeze(att.coalesce().values(), dim=-1), 0, 1),
                    index[:, 0])
                new_features = torch.transpose(new_features, 0, 1)
                features_list.append(new_features)

            if self.attn_heads_reduction == 'concat':
                features = torch.cat(features_list)

            features = self.activation(features)
            outputs.append(features)

        outputs = torch.cat(outputs, dim=1)
        proxy_att = torch.matmul(F.normalize(outputs, dim=-1),
                                 torch.transpose(F.normalize(self.proxy, dim=-1), 0, 1))
        proxy_att = F.softmax(proxy_att, dim=-1)  # eq.3
        proxy_feature = outputs - torch.matmul(proxy_att, self.proxy)

        if self.use_bias:
            gate_rate = F.sigmoid(torch.matmul(proxy_feature, self.gate_kernel) + self.bias)
        else:
            gate_rate = F.sigmoid(torch.matmul(proxy_feature, self.gate_kernel))
        outputs = gate_rate * outputs + (1 - gate_rate) * proxy_feature
        return outputs
