import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.init as init
import torch.nn.functional as F


class GraphConvolution(nn.Module):

    def __init__(self, input_dim, output_dim, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout

        self.act = act

        self.weight = Parameter(torch.FloatTensor(input_dim, output_dim))

        self.reset_parameters()

    def reset_parameters(self):

        init.kaiming_uniform_(self.weight)



    def forward(self, input, adj):

        input = F.dropout(input, self.dropout)

        support = torch.mm(input, self.weight)

        output = torch.spmm(adj, support)
        output = self.act(output)

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.input_dim) + ' -> ' \
               + str(self.output_dim) + ')'


class InnerProductDecoder(nn.Module):

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj



class MultiplyLayer(nn.Module):

    def __init__(self, num_A, num_B, act=F.relu):
        super(MultiplyLayer, self).__init__()


        self.act = act
        self.weight_A = Parameter(torch.FloatTensor(num_A, num_A))
        self.weight_B = Parameter(torch.FloatTensor(num_B, num_B))
        self.bias = Parameter(torch.zeros(num_B).type(torch.FloatTensor))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight_A)
        init.kaiming_uniform_(self.weight_B)
        init.zeros_(self.bias)

    def forward(self, inputs):
        express, adj_A, adj_B = inputs

        output_A = torch.mul(self.weight_A, adj_A)
        output_B = torch.mul(self.weight_B, adj_B)

        output_ax = torch.mm(output_A, express)
        output_xb = torch.mm(express, output_B)
        output_axb = torch.mm(torch.mm(output_A, express), output_B)

        output = (torch.add(torch.add(output_ax, output_xb), output_axb))
        output = torch.add(output, self.bias)

        return self.act(output)


class GAT(nn.Module):


    def __init__(self, num_of_layers, num_heads_per_layer, num_features_per_layer, add_skip_connection=True, bias=True,
                 dropout=0.6, log_attention_weights=False):
        super().__init__()
        assert num_of_layers == len(num_heads_per_layer) == len(num_features_per_layer) - 1, f'Enter valid arch params.'

        num_heads_per_layer = [1] + num_heads_per_layer

        gat_layers = []
        for i in range(num_of_layers):
            layer = GATLayer(
                num_in_features=num_features_per_layer[i],
                num_out_features=num_features_per_layer[i + 1],
                num_of_heads=num_heads_per_layer[i + 1],
                concat=False,
                activation=nn.ELU() if i < num_of_layers - 1 else None,
                dropout_prob=dropout,
                add_skip_connection=add_skip_connection,
                bias=bias,
                log_attention_weights=log_attention_weights
            )
            gat_layers.append(layer)

        self.gat_net = nn.Sequential(
            *gat_layers,
        )



    def forward(self, data):
        layer_outputs = []
        current_output = data

        for layer in self.gat_net:
            current_output = layer(current_output)
            layer_outputs.append(current_output)

        return layer_outputs[0][0], layer_outputs[1][0]

class GATLayer(torch.nn.Module):

    src_nodes_dim = 0
    trg_nodes_dim = 1

    nodes_dim = 0
    head_dim = 1

    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):

        super().__init__()

        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.concat = concat
        self.add_skip_connection = add_skip_connection
        self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        self.leakyReLU = nn.LeakyReLU(0.2)
        self.activation = activation

        self.dropout = nn.Dropout(p=dropout_prob)

        self.log_attention_weights = log_attention_weights
        self.attention_weights = None

        self.init_params()

    def forward(self, data):

        in_nodes_features, edge_index = data
        num_of_nodes = in_nodes_features.shape[self.nodes_dim]
        assert edge_index.shape[0] == 2, f'Expected edge index with shape=(2,E) got {edge_index.shape}'

        in_nodes_features = self.dropout(in_nodes_features)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads,
                                                                       self.num_out_features)

        nodes_features_proj = self.dropout(nodes_features_proj)

        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)

        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(scores_source, scores_target,
                                                                                           nodes_features_proj,
                                                                                           edge_index)
        scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)

        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[self.trg_nodes_dim],
                                                              num_of_nodes)

        attentions_per_edge = self.dropout(attentions_per_edge)

        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge

        out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index,
                                                      in_nodes_features, num_of_nodes)

        out_nodes_features = self.skip_concat_bias(attentions_per_edge, in_nodes_features,
                                                   out_nodes_features)
        return (out_nodes_features, edge_index)

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):

        scores_per_edge = scores_per_edge - scores_per_edge.max()
        exp_scores_per_edge = scores_per_edge.exp()

        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index,
                                                                                num_of_nodes)

        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)

        return attentions_per_edge.unsqueeze(-1)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)

        size = list(exp_scores_per_edge.shape)
        size[self.nodes_dim] = num_of_nodes
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)

        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)

        return neighborhood_sums.index_select(self.nodes_dim, trg_index)

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
        size = list(nodes_features_proj_lifted_weighted.shape)
        size[self.nodes_dim] = num_of_nodes
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)

        trg_index_broadcasted = self.explicit_broadcast(edge_index[self.trg_nodes_dim],
                                                        nodes_features_proj_lifted_weighted)
        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)

        return out_nodes_features

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):

        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]

        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)

        return scores_source, scores_target, nodes_features_matrix_proj_lifted

    def explicit_broadcast(self, this, other):
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        return this.expand_as(other)

    def init_params(self):

        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def skip_concat_bias(self, attention_coefficients, in_nodes_features, out_nodes_features):
        if self.log_attention_weights:
            self.attention_weights = attention_coefficients

        if self.add_skip_connection:
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.num_of_heads,
                                                                             self.num_out_features)

        if self.concat:
            out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)
        else:

            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)