import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot

from layers import MLP, Res


class Global_MessagePassing(MessagePassing):
    def __init__(self, config):
        super(Global_MessagePassing, self).__init__(flow=config.flow)
        self.dim = config.dim

        self.mlp_x1 = MLP([self.dim, self.dim])
        self.mlp_x2 = MLP([self.dim, self.dim])

        self.res1 = Res(self.dim)
        self.res2 = Res(self.dim)
        self.res3 = Res(self.dim)

        self.mlp_m = MLP([self.dim * 3, self.dim])
        self.W_edge_attr = nn.Linear(self.dim, self.dim, bias=False)

        self.mlp_out = MLP([self.dim, self.dim, self.dim, self.dim])
        self.W_out = nn.Linear(self.dim, 1)
        self.W = nn.Parameter(torch.Tensor(self.dim, 1))

        self.init()

    def init(self):
        glorot(self.W)

    def forward(self, x, edge_attr, edge_index):
        res_x = x
        x = self.mlp_x1(x)

        # print('41', x.max(), x.min())
        # Message Block
        x = x + self.propagate(edge_index, x=x, num_nodes=x.size(0), edge_attr=edge_attr)
        # print('42', x.max(), x.min())
        x = self.mlp_x2(x)
        # print('43', x.max(), x.min())

        # Update Block
        x = self.res1(x) + res_x
        # print('44', x.max(), x.min())
        x = self.res2(x)
        # print('45', x.max(), x.min())
        x = self.res3(x)
        # print('46', x.max(), x.min())

        out = self.mlp_out(x)
        # print('47', out.max(), out.min())
        att_score = out.matmul(self.W).unsqueeze(0)
        # print('48', att_score.max(), att_score.min())
        out = self.W_out(out).unsqueeze(0)
        # print('49', out.max(), out.min())

        return x, out, att_score

    def message(self, x_i, x_j, edge_attr, edge_index, num_nodes):
        m = torch.cat((x_i, x_j, edge_attr), -1)
        # print('50', m.max(), m.min())
        m = self.mlp_m(m)
        # print('51', m.max(), m.min())
        res = m * self.W_edge_attr(edge_attr)
        # print('52', res.max(), res.min())
        return res

    def update(self, aggr_out):

        return aggr_out
