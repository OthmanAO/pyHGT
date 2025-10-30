import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, uniform
from torch_geometric.utils import softmax
import math

class SpatialHGTConv(MessagePassing):
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, dropout=0.2, use_norm=True, use_spatial_encoding=True, **kwargs):
        super(SpatialHGTConv, self).__init__(node_dim=0, aggr='add', **kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_types = num_types
        self.num_relations = num_relations
        self.total_rel = num_types * num_relations * num_types
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.use_norm = use_norm
        self.use_spatial_encoding = use_spatial_encoding
        self.att = None
        
        # Linear layers for each node type
        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for t in range(num_types):
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))
        
        # Spatial relation parameters
        self.relation_pri = nn.Parameter(torch.ones(num_relations, self.n_heads))
        self.relation_att = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.skip = nn.Parameter(torch.ones(num_types))
        self.drop = nn.Dropout(dropout)
        
        # Spatial encoding for edge attributes
        if self.use_spatial_encoding:
            self.spatial_emb = SpatialEdgeEncoding(in_dim)
        
        glorot(self.relation_att)
        glorot(self.relation_msg)
        
    def forward(self, node_inp, node_type, edge_index, edge_type, edge_spatial):
        return self.propagate(edge_index, node_inp=node_inp, node_type=node_type,
                              edge_type=edge_type, edge_spatial=edge_spatial)

    def message(self, edge_index_i, node_inp_i, node_inp_j, node_type_i, node_type_j, edge_type, edge_spatial):
        """
        Spatial message passing with spatial edge attributes
        """
        data_size = edge_index_i.size(0)
        res_att = torch.zeros(data_size, self.n_heads).to(node_inp_i.device)
        res_msg = torch.zeros(data_size, self.n_heads, self.d_k).to(node_inp_i.device)
        
        for source_type in range(self.num_types):
            sb = (node_type_j == int(source_type))
            k_linear = self.k_linears[source_type]
            v_linear = self.v_linears[source_type]
            
            for target_type in range(self.num_types):
                tb = (node_type_i == int(target_type)) & sb
                q_linear = self.q_linears[target_type]
                
                for relation_type in range(self.num_relations):
                    idx = (edge_type == int(relation_type)) & tb
                    if idx.sum() == 0:
                        continue
                    
                    target_node_vec = node_inp_i[idx]
                    source_node_vec = node_inp_j[idx]
                    
                    # Apply spatial encoding to source representation
                    if self.use_spatial_encoding:
                        source_node_vec = self.spatial_emb(source_node_vec, edge_spatial[idx])
                    
                    # Step 1: Heterogeneous Mutual Attention
                    q_mat = q_linear(target_node_vec).view(-1, self.n_heads, self.d_k)
                    k_mat = k_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    k_mat = torch.bmm(k_mat.transpose(1, 0), self.relation_att[relation_type]).transpose(1, 0)
                    res_att[idx] = (q_mat * k_mat).sum(dim=-1) * self.relation_pri[relation_type] / self.sqrt_dk
                    
                    # Step 2: Heterogeneous Message Passing
                    v_mat = v_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    res_msg[idx] = torch.bmm(v_mat.transpose(1, 0), self.relation_msg[relation_type]).transpose(1, 0)
        
        # Softmax attention
        self.att = softmax(res_att, edge_index_i)
        res = res_msg * self.att.view(-1, self.n_heads, 1)
        del res_att, res_msg
        return res.view(-1, self.out_dim)

    def update(self, aggr_out, node_inp, node_type):
        """
        Target-specific aggregation with skip connections
        """
        aggr_out = F.gelu(aggr_out)
        res = torch.zeros(aggr_out.size(0), self.out_dim).to(node_inp.device)
        
        for target_type in range(self.num_types):
            idx = (node_type == int(target_type))
            if idx.sum() == 0:
                continue
            trans_out = self.drop(self.a_linears[target_type](aggr_out[idx]))
            
            # Skip connection with learnable weight
            alpha = torch.sigmoid(self.skip[target_type])
            if self.use_norm:
                res[idx] = self.norms[target_type](trans_out * alpha + node_inp[idx] * (1 - alpha))
            else:
                res[idx] = trans_out * alpha + node_inp[idx] * (1 - alpha)
        return res

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_relations={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)


class SpatialEdgeEncoding(nn.Module):
    """
    Spatial edge encoding to replace temporal encoding
    Encodes distance, direction, and other spatial attributes
    """
    def __init__(self, in_dim):
        super(SpatialEdgeEncoding, self).__init__()
        self.in_dim = in_dim
        self.distance_emb = nn.Linear(1, in_dim // 4)  # Distance encoding
        self.direction_emb = nn.Embedding(8, in_dim // 4)  # 8 directions
        self.relation_emb = nn.Linear(in_dim // 2, in_dim)  # Combine distance + direction
        self.lin = nn.Linear(in_dim, in_dim)
        
    def forward(self, x, spatial_attrs):
        """
        x: node features
        spatial_attrs: dict containing spatial attributes
        """
        # Extract spatial attributes
        distance = spatial_attrs.get('distance', 0.0)
        direction = spatial_attrs.get('direction', 'unknown')
        
        # Encode distance (normalize to 0-1)
        distance_norm = torch.tanh(torch.tensor(distance / 1000.0))  # Normalize by 1km
        distance_enc = self.distance_emb(distance_norm.unsqueeze(0))
        
        # Encode direction
        direction_map = {
            'north': 0, 'northeast': 1, 'east': 2, 'southeast': 3,
            'south': 4, 'southwest': 5, 'west': 6, 'northwest': 7
        }
        direction_idx = direction_map.get(direction, 0)
        direction_enc = self.direction_emb(torch.tensor(direction_idx))
        
        # Combine spatial encodings
        spatial_enc = torch.cat([distance_enc, direction_enc], dim=-1)
        spatial_enc = self.relation_emb(spatial_enc)
        
        # Add to node features
        return x + self.lin(spatial_enc)


class SpatialGeneralConv(nn.Module):
    """
    General spatial convolution wrapper
    """
    def __init__(self, conv_name, in_dim, out_dim, num_types, num_relations, n_heads, dropout=0.2, use_norm=True, use_spatial_encoding=True):
        super(SpatialGeneralConv, self).__init__()
        self.conv_name = conv_name
        self.base_conv = None
        
        if conv_name == 'spatial_hgt':
            self.base_conv = SpatialHGTConv(in_dim, out_dim, num_types, num_relations, n_heads, dropout, use_norm, use_spatial_encoding)
        elif conv_name == 'gcn':
            self.base_conv = GCNConv(in_dim, out_dim)
        elif conv_name == 'gat':
            self.base_conv = GATConv(in_dim, out_dim, heads=n_heads, dropout=dropout)
        else:
            raise NotImplementedError('Unknown conv: {}'.format(conv_name))
    
    def forward(self, meta_xs, node_type, edge_index, edge_type, edge_spatial):
        if self.conv_name in ['spatial_hgt']:
            return self.base_conv(meta_xs, node_type, edge_index, edge_type, edge_spatial)
        else:
            # For GCN/GAT, ignore spatial attributes
            return self.base_conv(meta_xs, edge_index)
