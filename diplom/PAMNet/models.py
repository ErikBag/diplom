import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn import global_mean_pool, global_add_pool, radius, knn
from torch_geometric.utils import remove_self_loops
import numpy as np

from layers import Global_MessagePassing, Local_MessagePassing, Local_MessagePassing_s, \
    BesselBasisLayer, SphericalBasisLayer, MLP

class Config(object):
    def __init__(self, dataset, dim, n_layer, cutoff_l, cutoff_g, flow='source_to_target'):
        self.dataset = dataset
        self.dim = dim
        self.n_layer = n_layer
        self.cutoff_l = cutoff_l
        self.cutoff_g = cutoff_g
        self.flow = flow

class PAMNet(nn.Module):
    def __init__(self, config: Config, num_spherical=7, num_radial=6, envelope_exponent=5):
        super(PAMNet, self).__init__()

        self.dataset = config.dataset
        self.dim = config.dim
        self.n_layer = config.n_layer
        self.cutoff_l = config.cutoff_l
        self.cutoff_g = config.cutoff_g

        self.embeddings = nn.Parameter(torch.ones((5, self.dim)))
        self.init_linear = nn.Linear(18, self.dim, bias=False)

        self.rbf_g = BesselBasisLayer(16, self.cutoff_g, envelope_exponent)
        self.rbf_l = BesselBasisLayer(16, self.cutoff_l, envelope_exponent)
        self.sbf = SphericalBasisLayer(num_spherical, num_radial, self.cutoff_l, envelope_exponent)

        self.mlp_rbf_g = MLP([16, self.dim])
        self.mlp_rbf_l = MLP([16, self.dim])    
        self.mlp_sbf1 = MLP([num_spherical * num_radial, self.dim])
        self.mlp_sbf2 = MLP([num_spherical * num_radial, self.dim])

        self.global_layer = torch.nn.ModuleList()
        for _ in range(config.n_layer):
            self.global_layer.append(Global_MessagePassing(config))

        self.local_layer = torch.nn.ModuleList()
        for _ in range(config.n_layer):
            self.local_layer.append(Local_MessagePassing(config))

        self.softmax = nn.Softmax(dim=-1)

        self.init()

    def init(self):
        stdv = math.sqrt(3)
        self.embeddings.data.uniform_(-stdv, stdv)

    def get_edge_info(self, edge_index, pos):
        edge_index, _ = remove_self_loops(edge_index)
        j, i = edge_index
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
        return edge_index, dist

    def indices(self, edge_index, num_nodes):
        row, col = edge_index

        value = torch.arange(row.size(0), device=row.device)
        adj_t = SparseTensor(row=col, col=row, value=value,
                             sparse_sizes=(num_nodes, num_nodes))
        adj_t_row = adj_t[row]
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

        idx_i = col.repeat_interleave(num_triplets)
        idx_j = row.repeat_interleave(num_triplets)
        idx_k = adj_t_row.storage.col()
        mask = idx_i != idx_k  # Remove i == k triplets.
        idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

        idx_kj = adj_t_row.storage.value()[mask]
        idx_ji = adj_t_row.storage.row()[mask]
        adj_t_col = adj_t[col]

        num_pairs = adj_t_col.set_value(None).sum(dim=1).to(torch.long)
        idx_i_pair = row.repeat_interleave(num_pairs)
        idx_j1_pair = col.repeat_interleave(num_pairs)
        idx_j2_pair = adj_t_col.storage.col()

        mask_j = idx_i_pair != idx_j2_pair  # Remove j == j' triplets.
        idx_i_pair, idx_j1_pair, idx_j2_pair = idx_i_pair[mask_j], idx_j1_pair[mask_j], idx_j2_pair[mask_j]

        idx_ji_pair = adj_t_col.storage.row()[mask_j]
        idx_jj_pair = adj_t_col.storage.value()[mask_j]

        return idx_i, idx_j, idx_k, idx_kj, idx_ji, idx_i_pair, idx_j1_pair, idx_j2_pair, idx_jj_pair, idx_ji_pair

    def forward(self, data):
        # print('\n\n\n new_forward')
        # print('1', data)
        x_raw = data.x
        batch = data.batch

        x_raw = x_raw.unsqueeze(-1) if x_raw.dim() == 1 else x_raw
        x = self.init_linear(x_raw[:, 3:-1])
        pos = x_raw[:,:3].contiguous()
        # print('pos', pos)

        # Indices for computing energy difference
        pos_index = torch.ones_like(pos[:, 0])
        neg_index = torch.ones_like(pos[:, 0]) * (-1.0)
        all_index = torch.where(pos[:, 0] > 0, neg_index, pos_index)

        # Compute pairwise distances in global layer
        row, col = radius(pos, pos, self.cutoff_g, batch, batch, max_num_neighbors=1000)
        edge_index_g = torch.stack([row, col], dim=0)
        edge_index_g, dist_g = self.get_edge_info(edge_index_g, pos)
        cut = self.cutoff_g
        while edge_index_g.numel() == 0 or dist_g.numel() == 0:
            # print("\n\n edge_index_g-dist_g  error \n\n")
            cut*=2
            row, col = radius(pos, pos, cut, batch, batch, max_num_neighbors=1000)
            edge_index_g = torch.stack([row, col], dim=0)
            edge_index_g, dist_g = self.get_edge_info(edge_index_g, pos)
        # print('edge_index_g', edge_index_g.shape)
        # print('3', dist_g[:10])
        dist_g = torch.clamp(dist_g, min=0.1)
        # Compute pairwise distances in local layer
        tensor_l = torch.ones_like(dist_g, device=dist_g.device) * self.cutoff_l
        mask_l = dist_g <= tensor_l
        edge_index_l = edge_index_g[:, mask_l]
        edge_index_l, dist_l = self.get_edge_info(edge_index_l, pos)
        cut = self.cutoff_l
        while edge_index_l.numel() == 0 or dist_l.numel() == 0:
            # print("\n\n edge_index_g-dist_g  error \n\n")
            cut*=2
            tensor_l = torch.ones_like(dist_g, device=dist_g.device) * cut
            mask_l = dist_g <= tensor_l
            edge_index_l = edge_index_g[:, mask_l]
            edge_index_l, dist_l = self.get_edge_info(edge_index_l, pos)
        # print('4', edge_index_l[:,:10])
        # print('5',dist_l[:10])
        dist_l = torch.clamp(dist_l, min=0.01)
        

        idx_i, idx_j, idx_k, idx_kj, idx_ji, idx_i_pair, idx_j1_pair, idx_j2_pair, idx_jj_pair, idx_ji_pair = self.indices(edge_index_l, num_nodes=x.size(0))
        # print('6', idx_i, idx_j, idx_k, idx_kj, idx_ji, idx_i_pair, idx_j1_pair, idx_j2_pair, idx_jj_pair, idx_ji_pair)
        # Compute two-hop angles in local layer
        pos_ji, pos_kj = pos[idx_j] - pos[idx_i], pos[idx_k] - pos[idx_j]
        a = (pos_ji * pos_kj).sum(dim=-1)
        b = torch.cross(pos_ji, pos_kj, dim=-1).norm(dim=-1)
        angle2 = torch.atan2(b, a)

        # print('7', a, b, angle2)

        # Compute one-hop angles in local layer
        pos_i_pair = pos[idx_i_pair]
        pos_j1_pair = pos[idx_j1_pair]
        pos_j2_pair = pos[idx_j2_pair]
        pos_ji_pair, pos_jj_pair = pos_j1_pair - pos_i_pair, pos_j2_pair - pos_j1_pair
        a = (pos_ji_pair * pos_jj_pair).sum(dim=-1)
        b = torch.cross(pos_ji_pair, pos_jj_pair, dim=-1).norm(dim=-1)
        angle1 = torch.atan2(b, a)

        # print('7', a, b, angle1)

        # print('dist_g', dist_g.max(), dist_g.min())

        # Get rbf and sbf embeddings
        rbf_l = self.rbf_l(dist_l)
        rbf_g = self.rbf_g(dist_g)
        sbf1 = self.sbf(dist_l, angle1, idx_jj_pair)
        sbf2 = self.sbf(dist_l, angle2, idx_kj)

        # print('13', rbf_l.max(), rbf_l.min())
        # print('13.5', rbf_g.max(), rbf_g.min())
        # print('14', sbf1.max(), sbf1.min())
        # print('14.5', sbf2.max(), sbf2.min())

        edge_attr_rbf_l = self.mlp_rbf_l(rbf_l)
        edge_attr_rbf_g = self.mlp_rbf_g(rbf_g)
        edge_attr_sbf1 = self.mlp_sbf1(sbf1)
        edge_attr_sbf2 = self.mlp_sbf2(sbf2)
        # print('15', edge_attr_rbf_l.max(), edge_attr_rbf_l.min())
        # print('15.5', edge_attr_rbf_g.max(), edge_attr_rbf_g.min())
        # print('16', edge_attr_sbf1.max(), edge_attr_sbf1.min())
        # print('16.5', edge_attr_sbf2.max(), edge_attr_sbf2.min())

        # Message Passing Modules
        out_global = []
        out_local = []
        att_score_global = []
        att_score_local = []
        for layer in range(self.n_layer):
            # print('x', x.max(), x.min())
            # print('earg', edge_attr_rbf_g.max(), edge_attr_rbf_g.min())
            # print(f'{layer} Всего памяти занято20: {end_mem/(1024**3):.2f} GB')
            x, out_g, att_score_g = self.global_layer[layer](x, edge_attr_rbf_g, edge_index_g)
            out_global.append(out_g)
            att_score_global.append(att_score_g)
            # print('x', x.max(), x.min())
            # print('out_g', out_g.max(), out_g.min())
            # print('att_score_g', att_score_g.max(), att_score_g.min())

            # print('earl', edge_attr_rbf_l.max(), edge_attr_rbf_l.min())
            # print('eas2', edge_attr_sbf2.max(), edge_attr_sbf2.min())
            # print('eas1', edge_attr_sbf1.max(), edge_attr_sbf1.min())
            x, out_l, att_score_l = self.local_layer[layer](x, edge_attr_rbf_l, edge_attr_sbf2, edge_attr_sbf1, \
                                                    idx_kj, idx_ji, idx_jj_pair, idx_ji_pair, edge_index_l)
            out_local.append(out_l)
            att_score_local.append(att_score_l)
            # print('x', x.max(), x.min())
            # print('out_l', out_l.max(), out_l.min())
            # print('att_score_l', att_score_l.max(), att_score_l.min())
        # Fusion Module
        att_score = torch.cat((torch.cat(att_score_global, 0), torch.cat(att_score_local, 0)), -1)
        att_score = F.leaky_relu(att_score, 0.2)
        att_weight = self.softmax(att_score)
        out = torch.cat((torch.cat(out_global, 0), torch.cat(out_local, 0)), -1)
        # print('9', out.max(), out.min())
        out = (out * att_weight).sum(dim=-1)
        # print('10', out.max(), out.min())
        out = out.sum(dim=0).unsqueeze(-1)
        # print('11', out.max(), out.min())

        out = out * all_index.unsqueeze(-1)
        # print('17', out.shape, out.max(), out.min())
        # print('18', out)
        out = global_add_pool(out, batch)

        # # print('12', out, '\n\n\n')
        # if torch.isnan(out).any() or np.abs(out.item()) > 10000:
        #     # print("out is nan")
        #     exit()
        return out.view(-1)


class PAMNet_s(nn.Module):
    def __init__(self, config: Config, num_spherical=7, num_radial=6, envelope_exponent=5):
        super(PAMNet_s, self).__init__()

        self.dataset = config.dataset
        self.dim = config.dim
        self.n_layer = config.n_layer
        self.cutoff_l = config.cutoff_l
        self.cutoff_g = config.cutoff_g

        self.embeddings = nn.Parameter(torch.ones((5, self.dim)))

        self.rbf_g = BesselBasisLayer(16, self.cutoff_g, envelope_exponent)
        self.rbf_l = BesselBasisLayer(16, self.cutoff_l, envelope_exponent)
        self.sbf = SphericalBasisLayer(num_spherical, num_radial, self.cutoff_l, envelope_exponent)

        self.mlp_rbf_g = MLP([16, self.dim])
        self.mlp_rbf_l = MLP([16, self.dim])    
        self.mlp_sbf = MLP([num_spherical * num_radial, self.dim])

        self.global_layer = torch.nn.ModuleList()
        for _ in range(config.n_layer):
            self.global_layer.append(Global_MessagePassing(config))

        self.local_layer = torch.nn.ModuleList()
        for _ in range(config.n_layer):
            self.local_layer.append(Local_MessagePassing_s(config))

        self.softmax = nn.Softmax(dim=-1)

        self.init()

    def init(self):
        stdv = math.sqrt(3)
        self.embeddings.data.uniform_(-stdv, stdv)

    def indices(self, edge_index, num_nodes):
        row, col = edge_index

        value = torch.arange(row.size(0), device=row.device)
        adj_t = SparseTensor(row=col, col=row, value=value,
                             sparse_sizes=(num_nodes, num_nodes))
        
        adj_t_col = adj_t[col]

        num_pairs = adj_t_col.set_value(None).sum(dim=1).to(torch.long)
        idx_i_pair = row.repeat_interleave(num_pairs)
        idx_j1_pair = col.repeat_interleave(num_pairs)
        idx_j2_pair = adj_t_col.storage.col()

        mask_j = idx_j1_pair != idx_j2_pair  # Remove j == j' triplets.
        idx_i_pair, idx_j1_pair, idx_j2_pair = idx_i_pair[mask_j], idx_j1_pair[mask_j], idx_j2_pair[mask_j]

        idx_ji_pair = adj_t_col.storage.row()[mask_j]
        idx_jj_pair = adj_t_col.storage.value()[mask_j]

        return idx_i_pair, idx_j1_pair, idx_j2_pair, idx_jj_pair, idx_ji_pair

    def forward(self, data):
        
        x_raw = data.x
        edge_index_l = data.edge_index
        pos = data.pos
        batch = data.batch
        x = torch.index_select(self.embeddings, 0, x_raw.long())
        
        # Compute pairwise distances in local layer
        edge_index_l, _ = remove_self_loops(edge_index_l)
        j_l, i_l = edge_index_l
        dist_l = (pos[i_l] - pos[j_l]).pow(2).sum(dim=-1).sqrt()

        # Compute pairwise distances in global layer
        row, col = radius(pos, pos, self.cutoff_g, batch, batch, max_num_neighbors=500)
        edge_index_g = torch.stack([row, col], dim=0)
        edge_index_g, _ = remove_self_loops(edge_index_g)
        j_g, i_g = edge_index_g
        dist_g = (pos[i_g] - pos[j_g]).pow(2).sum(dim=-1).sqrt()

        idx_i_pair, idx_j1_pair, idx_j2_pair, idx_jj_pair, idx_ji_pair = self.indices(edge_index_l, num_nodes=x.size(0))

        # Compute one-hop angles in local layer
        pos_i_pair = pos[idx_i_pair]
        pos_j1_pair = pos[idx_j1_pair]
        pos_j2_pair = pos[idx_j2_pair]
        pos_ji_pair, pos_jj_pair = pos_j1_pair - pos_i_pair, pos_j2_pair - pos_j1_pair
        a = (pos_ji_pair * pos_jj_pair).sum(dim=-1)
        b = torch.cross(pos_ji_pair, pos_jj_pair, dim=-1).norm(dim=-1)
        angle = torch.atan2(b, a)

        # Get rbf and sbf embeddings
        rbf_l = self.rbf_l(dist_l)
        rbf_g = self.rbf_g(dist_g)
        sbf = self.sbf(dist_l, angle, idx_jj_pair)

        edge_attr_rbf_l = self.mlp_rbf_l(rbf_l)
        edge_attr_rbf_g = self.mlp_rbf_g(rbf_g)
        edge_attr_sbf = self.mlp_sbf(sbf)

        # Message Passing Modules
        out_global = []
        out_local = []
        att_score_global = []
        att_score_local = []
        
        for layer in range(self.n_layer):
            x, out_g, att_score_g = self.global_layer[layer](x, edge_attr_rbf_g, edge_index_g)
            out_global.append(out_g)
            att_score_global.append(att_score_g)
            
            x, out_l, att_score_l = self.local_layer[layer](x, edge_attr_rbf_l, edge_attr_sbf, \
                                                            idx_jj_pair, idx_ji_pair, edge_index_l)
            out_local.append(out_l)
            att_score_local.append(att_score_l)
        
        # Fusion Module
        att_score = torch.cat((torch.cat(att_score_global, 0), torch.cat(att_score_local, 0)), -1)
        att_score = F.leaky_relu(att_score, 0.2)
        att_weight = self.softmax(att_score)

        out = torch.cat((torch.cat(out_global, 0), torch.cat(out_local, 0)), -1)
        out = (out * att_weight).sum(dim=-1)
        out = out.sum(dim=0).unsqueeze(-1)
        out = global_add_pool(out, batch)

        return out.view(-1)