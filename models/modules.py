
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_softmax
from models.heatmap import *

def coord2radial(edge_index, coord):
    row, col = edge_index
    coord_diff = coord[row] - coord[col]  # [n_edge, n_channel, d]
    radial = torch.bmm(coord_diff, coord_diff.transpose(-1, -2))  # [n_edge, n_channel, n_channel]
    # normalize radial
    radial = F.normalize(radial, dim=0)  # [n_edge, n_channel, n_channel]
    return radial, coord_diff


def sequential_and(*tensors):
    res = tensors[0]
    for mat in tensors[1:]:
        res = torch.logical_and(res, mat)
    return res


def sequential_or(*tensors):
    res = tensors[0]
    for mat in tensors[1:]:
        res = torch.logical_or(res, mat)
    return res


def unsorted_segment_sum(data, segment_ids, num_segments):
    expand_dims = tuple(data.shape[1:])
    result_shape = (num_segments,) + expand_dims
    for _ in expand_dims:
        segment_ids = segment_ids.unsqueeze(-1)
    segment_ids = segment_ids.expand(-1, *expand_dims)
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    expand_dims = tuple(data.shape[1:])
    result_shape = (num_segments,) + expand_dims
    for _ in expand_dims:
        segment_ids = segment_ids.unsqueeze(-1)
    segment_ids = segment_ids.expand(-1, *expand_dims)
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


class PositionEncoder(nn.Module):
    def __init__(self, input_dim=16, pos_dim=32, num_freq_bands=32):
        super().__init__()
        self.radial_proj = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 3)
        )


        self.frequencies = nn.Parameter(
            torch.logspace(-1, 2, num_freq_bands),
            requires_grad=False
        )


        self.proj = nn.Sequential(
            nn.Linear(3 * num_freq_bands * 2 + 4, pos_dim),
            nn.SiLU(),
            nn.Linear(pos_dim, pos_dim)
        )

    def forward(self, radial):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            batch_size = radial.size(0)
            coord_diff = self.radial_proj(radial.view(batch_size, -1))

            distance = torch.norm(coord_diff, dim=1, keepdim=True)
            direction = coord_diff / (distance + 1e-8)

            scaled_coords = coord_diff.unsqueeze(-1) * self.frequencies
            sin_enc = torch.sin(scaled_coords)
            cos_enc = torch.cos(scaled_coords)

        sin_flat = sin_enc.flatten(start_dim=1)
        cos_flat = cos_enc.flatten(start_dim=1)
        enc = torch.cat([sin_flat, cos_flat, distance, direction], dim=1)

        return self.proj(enc)

class GAMPNN(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, n_channel, dropout=0.1, edges_in_d=1, edge_type=8):
        # input_nf256 output_nf256 hidden_nf256 n_channel4 edges_in_d104
        super(GAMPNN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.coord_mlp = nn.ModuleList()
        self.Geo_mlp = nn.ModuleList()

        self.position_encoder = PositionEncoder(input_dim=4 * 4)
        self.node_interaction = nn.Sequential(
            nn.Linear(2 * input_nf, hidden_nf),
            nn.SiLU()
        )

        total_input_dim = hidden_nf + 32 + edges_in_d
        self.message_mlp = nn.Sequential(
            nn.Linear(total_input_dim, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU()
        )

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_nf + edges_in_d + hidden_nf + 1, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, edges_in_d))

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, output_nf))

        self.position_encoder = PositionEncoder()
        self.node_interaction = nn.Sequential(
            nn.Linear(2 * input_nf, hidden_nf),
            nn.SiLU()
        )

        for _ in range(edge_type):
            self.Geo_mlp.append(nn.Linear(input_nf, input_nf, bias=False))

            layer = nn.Linear(hidden_nf, n_channel, bias=False)
            torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

            self.coord_mlp.append(nn.Sequential(
                nn.Linear(hidden_nf, hidden_nf),
                nn.SiLU(),
                layer
            ))

        num_channels = 4
        self.epsilon = nn.Parameter(torch.tensor(2.6))
        self.sigma = nn.Parameter(torch.tensor(3.6))
        self.att_sigma = nn.Parameter(torch.tensor(16.6))
        # self.linear_q = nn.Sequential(
        #     nn.Linear(hidden_nf , output_nf),
        #     nn.SiLU()
        # )
        self.linear_k = nn.Sequential(
            nn.Linear(hidden_nf + num_channels ** 2 + edges_in_d, hidden_nf),
            nn.SiLU()
        )
        self.linear_q = nn.Linear(hidden_nf, hidden_nf)
        #self.linear_k = nn.Linear(hidden_nf + num_channels ** 2 + edges_in_d, hidden_nf)
        self.linear_v = nn.Linear(hidden_nf + num_channels ** 2 + edges_in_d, hidden_nf)
        # self.linear_v = nn.Sequential(
        #     nn.Linear(hidden_nf + num_channels ** 2 + edges_in_d, hidden_nf),
        #     nn.SiLU()
        # )


    def message_model(self, source, target, radial, edge_attr):

        pos_encoding = self.position_encoder(radial)
        node_interaction = self.node_interaction(
            torch.cat([source, target], dim=1)
        )

        out = torch.cat([
            node_interaction,
            pos_encoding,
            edge_attr
        ], dim=1)


        out = self.message_mlp(out)
        return self.dropout(out)

    def att_module(self, h, edge_list, radial, edge_attr_list, coord_diff_list):

        attentions = []
        values = []
        try:
            row, col = edge_list[0], edge_list[1]
        except Exception as e:
            print(f"Error accessing edge_tensor elements: {e}")

        row = row.flatten()
        col = col.flatten()

        try:
            source = h[row]
            target = h[col]
        except Exception as e:
            print(f"Error slicing h for source and target: {e}")

        # Query/Key/Value generation
        try:
            q = self.linear_q(source)
            n_channel = 4
            # radial_expanded = radial.unsqueeze(0)
            radial_flat = radial.reshape(radial.shape[0], n_channel * n_channel)
        except Exception as e:
            print(f"Error during Q/K/V generation: {e}")

        try:
            if edge_attr_list is not None:
                n_edge = edge_attr_list.shape[0]
                radial_flat = radial_flat.expand(n_edge, -1)
                target = target.expand(n_edge, -1)
                target_feat = torch.cat([radial_flat, target, edge_attr_list], dim=1)
            else:
                target_feat = torch.cat([radial_flat, target], dim=1)
        except RuntimeError as e:
            print(f"Error concatenating features: {e}")

        try:
            k = self.linear_k(target_feat)
            v = self.linear_v(target_feat)
        except Exception as e:
            print(f"Error during KV generation: {e}")

        # Spatial Gaussian Position Bias
        try:
            ca_coord_diff = coord_diff_list[:, 1, :]
            distance = torch.norm(ca_coord_diff, dim=-1)  # [n_node, 2]
            absolute_pos_bias = torch.exp(-distance ** 2 / (2 * self.att_sigma ** 2))
        except Exception as e:
            print(f"Error generating positional bias: {e}")

        try:
            attention_score = (q * k).sum(dim=-1) + absolute_pos_bias  # [n_edge]
        except Exception as e:
            print(f"Error calculating attention score: {e}")

        # Apply softmax along the outgoing edges of each node (row)
        try:
            alpha = scatter_softmax(attention_score, row, dim=0)  # [n_edge]
            attentions.append(alpha)
            values.append(v)
        except Exception as e:
            print(f"Error during softmax calculation: {e}")

        return attentions, values

    def node_model(self, x, edge_list, edge_feat_list, att_weight_list, att_v):
        agg = self.Geo_mlp[0](unsorted_segment_sum(edge_feat_list[0], edge_list[0][0], num_segments=x.size(0)))
        for i in range(1, len(edge_list)):
            agg += self.Geo_mlp[i](
                unsorted_segment_sum(edge_feat_list[i] * att_weight_list[i], edge_list[i][0],
                                     num_segments=x.size(0)))

        agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        out = self.dropout(out)
        out = x + out

        return out

    def coord_model(self, coord, edge_list, edge_feat_list, coord_diff_list):
        tran_list = []
        row_list = []

        for i in range(len(edge_list)):
            edge_feat = edge_feat_list[i]
            coord_diff = coord_diff_list[i]
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                scale = self.coord_mlp[i](edge_feat)
                trans = coord_diff * scale.unsqueeze(-1)
            tran_list.append(trans.float())
            row_list.append(edge_list[i][0])


        trans_tensor = torch.cat(tran_list, dim=0)  # [total_edges, n_channel, d]
        row_indices = torch.cat(row_list, dim=0)  # [total_edges]

        agg = unsorted_segment_mean(trans_tensor, row_indices, coord.size(0))
        coord = coord + agg

        return coord

    def lj_potential(self, ca_diff):
        r = torch.norm(ca_diff, dim=1, keepdim=True)  # [E,1]
        r = torch.clamp(r, min=0.1) + 1e-6
        energy = 4 * self.epsilon * ((self.sigma / r) ** 12 - (self.sigma / r) ** 6)
        return energy

    def edge_model(self, h, edge_list, edge_attr, coord_diff_list):
        m = []

        for i in range(len(edge_list)):
            row, col = edge_list[i]
            ca_diff = coord_diff_list[i][:, 1, :]
            energy = self.lj_potential(ca_diff)
            energy = (energy - energy.mean()) / (energy.std() + 1e-6)
            out = torch.cat([h[row], edge_attr[i], h[col], energy], dim=1)
            out = self.edge_mlp(out)
            m.append(out)
        return m

    def forward(self, h, coord, edge_attr, edge_list, segment_idx, segment_ids):

        edge_feat_list = []
        coord_diff_list = []

        att_weight_list = []
        flat_att_weight = []
        att_v = []

        for i in range(len(edge_list)):
            radial, coord_diff = coord2radial(edge_list[i], coord)
            coord_diff_list.append(coord_diff)

            att_weight, v = self.att_module(h, edge_list[i], radial, edge_attr[i], coord_diff_list[i])

            flat_att_weight.append(att_weight[0].unsqueeze(-1))
            att_weight_list.append(att_weight[0].unsqueeze(-1))
            att_v.append(v[0])

            row, col = edge_list[i]
            edge_feat = self.message_model(h[row], h[col], radial, edge_attr[i])
            edge_feat_list.append(edge_feat)

        x = self.coord_model(coord, edge_list, edge_feat_list, coord_diff_list)
        h = self.node_model(h, edge_list, edge_feat_list, att_weight_list, att_v)
        m = self.edge_model(h, edge_list, edge_attr, coord_diff_list)

        return h, x, m, flat_att_weight


class GAEGNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, n_channel, n_layers=4, dropout=0.1, node_feats_dim=0,
                 edge_feats_dim=1): # in_node_nf32 hidden_nf256 out_node_nf20 n_channel4 node_feats_dim94 edge_feats_dim104
        super().__init__()
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)

        self.linear_in = nn.Linear(in_node_nf + node_feats_dim, hidden_nf)
        self.linear_out = nn.Linear(hidden_nf, out_node_nf)

        for i in range(n_layers):
            self.add_module(f'layer_{i}', GAMPNN(hidden_nf, hidden_nf, hidden_nf, n_channel, dropout=dropout,
                                                       edges_in_d=edge_feats_dim))

    def forward(self, h, x, edges_list, edge_feats_list, node_feats, interface_only, segment_idx, segment_ids):
        h = torch.cat((h, node_feats), 1)
        h = self.linear_in(h)
        h = self.dropout(h)

        m = edge_feats_list
        for i in range(self.n_layers):
            if interface_only == 0:
                h, x, m = self._modules[f'layer_{i}'](h, x, m, edges_list)
            else:
                h, x, m, att = self._modules[f'layer_{i}'](h, x, m, edges_list, segment_idx, segment_ids)

        out = self.dropout(h)
        out = self.linear_out(out)

        return out, x, h