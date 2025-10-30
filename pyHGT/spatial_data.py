import json, os
import math, copy, time
import numpy as np
from collections import defaultdict
import pandas as pd
from .utils import *
import torch
import math
from tqdm import tqdm
import dill
from functools import partial
import multiprocessing as mp

class SpatialGraph():
    def __init__(self):
        super(SpatialGraph, self).__init__()
        '''
            Spatial version of Graph class
            node_forward: name -> node_id
            node_bacward: node_id -> feature_dict
            node_feature: a DataFrame containing all features
        '''
        self.node_forward = defaultdict(lambda: {})
        self.node_bacward = defaultdict(lambda: [])
        self.node_feature = defaultdict(lambda: [])

        '''
            edge_list: index the adjacency matrix by 
            <target_type, source_type, relation_type, target_id, source_id>
            Store spatial attributes instead of time
        '''
        self.edge_list = defaultdict( #target_type
                            lambda: defaultdict(  #source_type
                                lambda: defaultdict(  #relation_type
                                    lambda: defaultdict(  #target_id
                                        lambda: defaultdict( #source_id(
                                            lambda: dict # spatial attributes
                                        )))))
        self.spatial_attrs = {}
        
    def add_node(self, node):
        nfl = self.node_forward[node['type']]
        if node['id'] not in nfl:
            self.node_bacward[node['type']] += [node]
            ser = len(nfl)
            nfl[node['id']] = ser
            return ser
        return nfl[node['id']]
        
    def add_spatial_edge(self, source_node, target_node, relation_type=None, spatial_attrs=None, directed=True):
        """
        Add spatial edge with spatial attributes
        spatial_attrs: dict containing distance, direction, etc.
        """
        edge = [self.add_node(source_node), self.add_node(target_node)]
        
        # Store spatial attributes instead of time
        self.edge_list[target_node['type']][source_node['type']][relation_type][edge[1]][edge[0]] = spatial_attrs or {}
        
        if directed:
            # Add reverse edge with opposite direction
            reverse_attrs = spatial_attrs.copy() if spatial_attrs else {}
            if 'direction' in reverse_attrs:
                reverse_attrs['direction'] = self._reverse_direction(reverse_attrs['direction'])
            self.edge_list[source_node['type']][target_node['type']]['rev_' + relation_type][edge[0]][edge[1]] = reverse_attrs
        else:
            self.edge_list[source_node['type']][target_node['type']][relation_type][edge[0]][edge[1]] = spatial_attrs or {}
            
    def _reverse_direction(self, direction):
        """Reverse spatial direction"""
        direction_map = {
            'north': 'south', 'south': 'north',
            'east': 'west', 'west': 'east',
            'northeast': 'southwest', 'southwest': 'northeast',
            'northwest': 'southeast', 'southeast': 'northwest'
        }
        return direction_map.get(direction, direction)
        
    def update_node(self, node):
        nbl = self.node_bacward[node['type']]
        ser = self.add_node(node)
        for k in node:
            if k not in nbl[ser]:
                nbl[ser][k] = node[k]

    def get_meta_graph(self):
        types = self.get_types()
        metas = []
        for target_type in self.edge_list:
            for source_type in self.edge_list[target_type]:
                for r_type in self.edge_list[target_type][source_type]:
                    metas += [(target_type, source_type, r_type)]
        return metas
    
    def get_types(self):
        return list(self.node_feature.keys())

def spatial_feature_extractor(layer_data, graph):
    """
    Extract spatial features for nodes
    """
    feature = defaultdict(lambda: [])
    times = defaultdict(lambda: [])  # Keep for compatibility, but use spatial features
    indxs = defaultdict(lambda: [])
    texts = defaultdict(lambda: [])
    
    for _type in layer_data:
        for _key in layer_data[_type]:
            _ser = layer_data[_type][_key][0]
            node_data = graph.node_bacward[_type][_key]
            
            # Create spatial feature vector
            spatial_features = create_spatial_node_features(node_data)
            feature[_type] += [spatial_features]
            
            # Use a dummy time for compatibility (could be distance-based)
            times[_type] += [0]  # or use distance as "time"
            indxs[_type] += [_ser]
            texts[_type] += [node_data.get('name', _key)]
            
    return feature, times, indxs, texts

def create_spatial_node_features(node_data):
    """
    Create feature vector for spatial node
    """
    features = []
    
    # Basic spatial features
    features.extend([
        node_data.get('latitude', 0.0),
        node_data.get('longitude', 0.0),
        node_data.get('area', 0.0),
        node_data.get('population', 0.0),
        node_data.get('elevation', 0.0),
    ])
    
    # Node type encoding (one-hot)
    node_type = node_data.get('type', 'unknown')
    type_encoding = {
        'park': [1, 0, 0, 0, 0],
        'school': [0, 1, 0, 0, 0], 
        'building': [0, 0, 1, 0, 0],
        'street': [0, 0, 0, 1, 0],
        'unknown': [0, 0, 0, 0, 1]
    }
    features.extend(type_encoding.get(node_type, [0, 0, 0, 0, 1]))
    
    # Additional features
    features.extend([
        node_data.get('has_parking', 0),
        node_data.get('is_public', 0),
        node_data.get('rating', 0.0),
    ])
    
    return features

def spatial_sample_subgraph(graph, query_node, sampled_depth=3, sampled_number=50, max_distance=1000):
    """
    Sample subgraph based on spatial proximity instead of temporal
    """
    layer_data = defaultdict(lambda: {})
    budget = defaultdict(lambda: defaultdict(lambda: [0., 0]))  # [spatial_score, distance]
    new_layer_adj = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [])))
    
    # Add query node to layer_data
    query_type = query_node['type']
    query_id = query_node['id']
    layer_data[query_type][query_id] = [0, 0]  # [ser, distance]
    
    def add_spatial_budget(te, target_id, target_distance, layer_data, budget):
        """Add nodes to budget based on spatial relationships"""
        for source_type in te:
            tes = te[source_type]
            for relation_type in tes:
                if relation_type == 'self' or target_id not in tes[relation_type]:
                    continue
                adl = tes[relation_type][target_id]
                
                for source_id, spatial_attrs in adl.items():
                    if spatial_attrs is None:
                        continue
                        
                    distance = spatial_attrs.get('distance', float('inf'))
                    if distance > max_distance or source_id in layer_data[source_type]:
                        continue
                        
                    # Spatial scoring based on distance (closer = higher score)
                    spatial_score = 1.0 / (1.0 + distance / 100.0)  # Normalize distance
                    budget[source_type][source_id][0] += spatial_score
                    budget[source_type][source_id][1] = distance
    
    # Initialize budget with query node
    for _type in [query_type]:
        te = graph.edge_list[_type]
        for _id in [query_id]:
            add_spatial_budget(te, _id, 0, layer_data, budget)
    
    # Recursively sample based on spatial proximity
    for layer in range(sampled_depth):
        sts = list(budget.keys())
        for source_type in sts:
            te = graph.edge_list[source_type]
            keys = np.array(list(budget[source_type].keys()))
            
            if sampled_number > len(keys):
                sampled_ids = np.arange(len(keys))
            else:
                # Sample based on spatial score (proximity)
                score = np.array(list(budget[source_type].values()))[:, 0] ** 2
                score = score / np.sum(score)
                sampled_ids = np.random.choice(len(score), sampled_number, p=score, replace=False)
                
            sampled_keys = keys[sampled_ids]
            
            # Add sampled nodes
            for k in sampled_keys:
                layer_data[source_type][k] = [len(layer_data[source_type]), budget[source_type][k][1]]
            
            # Update budget
            for k in sampled_keys:
                add_spatial_budget(te, k, budget[source_type][k][1], layer_data, budget)
                budget[source_type].pop(k)
    
    # Prepare features and adjacency
    feature, times, indxs, texts = spatial_feature_extractor(layer_data, graph)
    
    edge_list = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [])))
    
    # Add self-loops
    for _type in layer_data:
        for _key in layer_data[_type]:
            _ser = layer_data[_type][_key][0]
            edge_list[_type][_type]['self'] += [[_ser, _ser]]
    
    # Reconstruct spatial adjacency
    for target_type in graph.edge_list:
        te = graph.edge_list[target_type]
        tld = layer_data[target_type]
        for source_type in te:
            tes = te[source_type]
            sld = layer_data[source_type]
            for relation_type in tes:
                tesr = tes[relation_type]
                for target_key in tld:
                    if target_key not in tesr:
                        continue
                    target_ser = tld[target_key][0]
                    for source_key in tesr[target_key]:
                        if source_key in sld:
                            source_ser = sld[source_key][0]
                            edge_list[target_type][source_type][relation_type] += [[target_ser, source_ser]]
    
    return feature, times, edge_list, indxs, texts

def to_spatial_torch(feature, time, edge_list, graph):
    """
    Transform spatial subgraph into PyTorch tensors
    """
    node_dict = {}
    node_feature = []
    node_type = []
    node_time = []  # Keep for compatibility
    edge_index = []
    edge_type = []
    edge_time = []  # Keep for compatibility, but could store distance
    
    node_num = 0
    types = graph.get_types()
    for t in types:
        node_dict[t] = [node_num, len(node_dict)]
        node_num += len(feature[t])

    for t in types:
        node_feature += list(feature[t])
        node_time += list(time[t])
        node_type += [node_dict[t][1] for _ in range(len(feature[t]))]
    
    edge_dict = {e[2]: i for i, e in enumerate(graph.get_meta_graph())}
    edge_dict['self'] = len(edge_dict)

    for target_type in edge_list:
        for source_type in edge_list[target_type]:
            for relation_type in edge_list[target_type][source_type]:
                for ii, (ti, si) in enumerate(edge_list[target_type][source_type][relation_type]):
                    tid, sid = ti + node_dict[target_type][0], si + node_dict[source_type][0]
                    edge_index += [[sid, tid]]
                    edge_type += [edge_dict[relation_type]]
                    # Use distance as "time" for compatibility
                    edge_time += [0]  # Could store actual distance here
    
    node_feature = torch.FloatTensor(node_feature)
    node_type = torch.LongTensor(node_type)
    edge_time = torch.LongTensor(edge_time)
    edge_index = torch.LongTensor(edge_index).t()
    edge_type = torch.LongTensor(edge_type)
    
    return node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict
