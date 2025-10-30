import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'pyHGT'))

from pyHGT.spatial_data import *
from pyHGT.spatial_model import *
from warnings import filterwarnings
filterwarnings("ignore")

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import json

parser = argparse.ArgumentParser(description='Training Spatial HGT for spatial queries')

# Dataset arguments
parser.add_argument('--data_dir', type=str, default='./spatial_dataset',
                    help='The address of spatial graph data.')
parser.add_argument('--model_dir', type=str, default='./spatial_model_save',
                    help='The address for storing the models.')
parser.add_argument('--task_name', type=str, default='spatial_query',
                    help='The name of the stored models.')
parser.add_argument('--cuda', type=int, default=0,
                    help='Available GPU ID')

# Model arguments
parser.add_argument('--conv_name', type=str, default='spatial_hgt',
                    choices=['spatial_hgt', 'gcn', 'gat'],
                    help='The name of GNN filter.')
parser.add_argument('--n_hid', type=int, default=400,
                    help='Number of hidden dimension')
parser.add_argument('--n_heads', type=int, default=8,
                    help='Number of attention head')
parser.add_argument('--n_layers', type=int, default=3,
                    help='Number of GNN layers')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout ratio')
parser.add_argument('--sample_depth', type=int, default=3,
                    help='Depth of spatial sampling')
parser.add_argument('--sample_width', type=int, default=50,
                    help='Width of spatial sampling per layer')
parser.add_argument('--max_distance', type=float, default=1000.0,
                    help='Maximum distance for spatial sampling (meters)')

# Training arguments
parser.add_argument('--optimizer', type=str, default='adamw',
                    choices=['adamw', 'adam', 'sgd'],
                    help='Optimizer to use.')
parser.add_argument('--data_percentage', type=float, default=1.0,
                    help='Percentage of training data to use')
parser.add_argument('--n_epoch', type=int, default=100,
                    help='Number of epoch to run')
parser.add_argument('--n_pool', type=int, default=4,
                    help='Number of process to sample subgraph')
parser.add_argument('--n_batch', type=int, default=32,
                    help='Number of batch (sampled graphs) per batch')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Number of nodes in a batch')
parser.add_argument('--clip', type=float, default=0.25,
                    help='Gradient Norm Clipping')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help='Weight decay')

# Query arguments
parser.add_argument('--query_type', type=str, default='nearest_neighbor',
                    choices=['nearest_neighbor', 'proximity', 'classification'],
                    help='Type of spatial query to train for')

args = parser.parse_args()

def create_sample_spatial_data():
    """
    Create sample spatial data for demonstration
    """
    graph = SpatialGraph()
    
    # Add sample nodes
    nodes = [
        {'id': 'central_park', 'type': 'park', 'latitude': 40.7829, 'longitude': -73.9654, 'area': 843, 'population': 0, 'is_public': 1},
        {'id': 'empire_state', 'type': 'building', 'latitude': 40.7484, 'longitude': -73.9857, 'area': 100, 'population': 0, 'is_public': 0},
        {'id': 'columbia_uni', 'type': 'school', 'latitude': 40.8075, 'longitude': -73.9626, 'area': 200, 'population': 30000, 'is_public': 1},
        {'id': 'times_square', 'type': 'building', 'latitude': 40.7580, 'longitude': -73.9855, 'area': 50, 'population': 0, 'is_public': 0},
        {'id': 'bryant_park', 'type': 'park', 'latitude': 40.7536, 'longitude': -73.9832, 'area': 9.6, 'population': 0, 'is_public': 1},
    ]
    
    for node in nodes:
        graph.add_node(node)
    
    # Add spatial relationships
    relationships = [
        ('central_park', 'empire_state', 'distance_400m', {'distance': 400, 'direction': 'south'}),
        ('central_park', 'columbia_uni', 'distance_800m', {'distance': 800, 'direction': 'north'}),
        ('empire_state', 'times_square', 'distance_300m', {'distance': 300, 'direction': 'west'}),
        ('bryant_park', 'times_square', 'distance_200m', {'distance': 200, 'direction': 'north'}),
        ('bryant_park', 'empire_state', 'distance_500m', {'distance': 500, 'direction': 'south'}),
    ]
    
    for source_id, target_id, relation, attrs in relationships:
        source_node = next(n for n in nodes if n['id'] == source_id)
        target_node = next(n for n in nodes if n['id'] == target_id)
        graph.add_spatial_edge(source_node, target_node, relation, attrs)
    
    return graph

def spatial_query_sample(seed, query_node, graph, sample_depth=3, sample_width=50, max_distance=1000):
    """
    Sample subgraph for spatial query
    """
    np.random.seed(seed)
    feature, times, edge_list, indxs, texts = spatial_sample_subgraph(
        graph, query_node, sample_depth, sample_width, max_distance
    )
    node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = to_spatial_torch(
        feature, times, edge_list, graph
    )
    return node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict

def prepare_spatial_data(pool, task_type='train', s_idx=0, n_batch=32, batch_size=256):
    """
    Prepare spatial data for training
    """
    jobs = []
    for i in range(s_idx, s_idx + n_batch):
        p = pool.apply_async(spatial_query_sample, args=(i, query_nodes[i % len(query_nodes)], graph, args.sample_depth, args.sample_width, args.max_distance))
        jobs.append(p)
    return jobs

def train_spatial_model():
    """
    Train spatial HGT model
    """
    # Create sample data
    graph = create_sample_spatial_data()
    
    # Get node types and relations
    node_types = list(graph.get_types())
    num_types = len(node_types)
    relations = graph.get_meta_graph()
    num_relations = len(relations)
    
    print(f"Node types: {node_types}")
    print(f"Relations: {relations}")
    print(f"Number of node types: {num_types}")
    print(f"Number of relations: {num_relations}")
    
    # Create model
    model = SpatialQueryModel(
        in_dim=13,  # Based on spatial feature extractor
        n_hid=args.n_hid,
        num_types=num_types,
        num_relations=num_relations,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_out=num_types,  # Classification into node types
        dropout=args.dropout,
        conv_name=args.conv_name
    )
    
    # Setup training
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Create sample query nodes for training
    global query_nodes
    query_nodes = [
        {'id': 'empire_state', 'type': 'building'},
        {'id': 'central_park', 'type': 'park'},
        {'id': 'columbia_uni', 'type': 'school'},
    ]
    
    # Training loop
    model.train()
    for epoch in range(args.n_epoch):
        total_loss = 0
        
        for batch_idx in range(args.n_batch):
            # Sample query node
            query_node = query_nodes[batch_idx % len(query_nodes)]
            
            # Sample subgraph
            try:
                node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = spatial_query_sample(
                    batch_idx, query_node, graph, args.sample_depth, args.sample_width, args.max_distance
                )
                
                # Move to device
                node_feature = node_feature.to(device)
                node_type = node_type.to(device)
                edge_time = edge_time.to(device)
                edge_index = edge_index.to(device)
                edge_type = edge_type.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                output = model(node_feature, node_type, edge_time, edge_index, edge_type, 'classification')
                
                # Create dummy labels (in real scenario, you'd have actual labels)
                labels = torch.randint(0, num_types, (node_type.size(0)).to(device)
                loss = F.nll_loss(output, labels)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                
                total_loss += loss.item()
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss / args.n_batch:.4f}")
    
    # Save model
    os.makedirs(args.model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.model_dir, f"{args.task_name}.pkl"))
    print(f"Model saved to {args.model_dir}/{args.task_name}.pkl")

if __name__ == '__main__':
    train_spatial_model()
