from .spatial_conv import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SpatialGNN(nn.Module):
    def __init__(self, in_dim, n_hid, num_types, num_relations, n_heads, n_layers, dropout=0.2, conv_name='spatial_hgt', prev_norm=False, last_norm=False, use_spatial_encoding=True):
        super(SpatialGNN, self).__init__()
        self.gcs = nn.ModuleList()
        self.num_types = num_types
        self.in_dim = in_dim
        self.n_hid = n_hid
        self.adapt_ws = nn.ModuleList()
        self.drop = nn.Dropout(dropout)
        
        # Type-specific adaptation layers
        for t in range(num_types):
            self.adapt_ws.append(nn.Linear(in_dim, n_hid))
        
        # GNN layers
        for l in range(n_layers - 1):
            self.gcs.append(SpatialGeneralConv(conv_name, n_hid, n_hid, num_types, num_relations, n_heads, dropout, use_norm=prev_norm, use_spatial_encoding=use_spatial_encoding))
        self.gcs.append(SpatialGeneralConv(conv_name, n_hid, n_hid, num_types, num_relations, n_heads, dropout, use_norm=last_norm, use_spatial_encoding=use_spatial_encoding))

    def forward(self, node_feature, node_type, edge_time, edge_index, edge_type):
        # Type-specific feature adaptation
        res = torch.zeros(node_feature.size(0), self.n_hid).to(node_feature.device)
        for t_id in range(self.num_types):
            idx = (node_type == int(t_id))
            if idx.sum() == 0:
                continue
            res[idx] = torch.tanh(self.adapt_ws[t_id](node_feature[idx]))
        
        meta_xs = self.drop(res)
        del res
        
        # Apply GNN layers
        for gc in self.gcs:
            meta_xs = gc(meta_xs, node_type, edge_index, edge_type, edge_time)
        return meta_xs


class SpatialClassifier(nn.Module):
    """
    Classifier for spatial node classification tasks
    """
    def __init__(self, n_hid, n_out):
        super(SpatialClassifier, self).__init__()
        self.n_hid = n_hid
        self.n_out = n_out
        self.linear = nn.Linear(n_hid, n_out)
        
    def forward(self, x):
        tx = self.linear(x)
        return torch.log_softmax(tx.squeeze(), dim=-1)
    
    def __repr__(self):
        return '{}(n_hid={}, n_out={})'.format(self.__class__.__name__, self.n_hid, self.n_out)


class SpatialMatcher(nn.Module):
    """
    Spatial matcher for proximity queries and link prediction
    """
    def __init__(self, n_hid):
        super(SpatialMatcher, self).__init__()
        self.left_linear = nn.Linear(n_hid, n_hid)
        self.right_linear = nn.Linear(n_hid, n_hid)
        self.sqrt_hd = math.sqrt(n_hid)
        self.cache = None
        
    def forward(self, x, y, infer=False, pair=False):
        """
        Match spatial nodes based on their representations
        """
        ty = self.right_linear(y)
        if infer:
            if self.cache is not None:
                tx = self.cache
            else:
                tx = self.left_linear(x)
                self.cache = tx
        else:
            tx = self.left_linear(x)
            
        if pair:
            res = (tx * ty).sum(dim=-1)
        else:
            res = torch.matmul(tx, ty.transpose(0, 1))
        return res / self.sqrt_hd
    
    def __repr__(self):
        return '{}(n_hid={})'.format(self.__class__.__name__, self.n_hid)


class SpatialQueryModel(nn.Module):
    """
    Complete model for spatial queries
    """
    def __init__(self, in_dim, n_hid, num_types, num_relations, n_heads, n_layers, n_out, dropout=0.2, conv_name='spatial_hgt'):
        super(SpatialQueryModel, self).__init__()
        self.gnn = SpatialGNN(in_dim, n_hid, num_types, num_relations, n_heads, n_layers, dropout, conv_name)
        self.classifier = SpatialClassifier(n_hid, n_out)
        self.matcher = SpatialMatcher(n_hid)
        
    def forward(self, node_feature, node_type, edge_time, edge_index, edge_type, query_type='classification'):
        """
        Forward pass for different query types
        """
        # Get node representations
        node_repr = self.gnn(node_feature, node_type, edge_time, edge_index, edge_type)
        
        if query_type == 'classification':
            return self.classifier(node_repr)
        elif query_type == 'matching':
            return node_repr  # Return representations for matching
        else:
            return node_repr
