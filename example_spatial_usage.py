#!/usr/bin/env python3
"""
Example usage of Spatial HGT for spatial queries

This script demonstrates how to use the spatial HGT implementation
for various spatial queries like finding nearest parks, adjacent buildings, etc.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'pyHGT'))

from pyHGT.spatial_data import SpatialGraph, create_sample_spatial_data
from spatial_query import SpatialQueryEngine, create_sample_spatial_graph
import torch

def main():
    print("=== Spatial HGT Example Usage ===\n")
    
    # 1. Create spatial graph
    print("1. Creating spatial graph...")
    graph = create_sample_spatial_graph()
    
    print(f"   - Node types: {list(graph.get_types())}")
    print(f"   - Relations: {graph.get_meta_graph()}")
    print(f"   - Total nodes: {sum(len(graph.node_bacward[t]) for t in graph.get_types())}")
    
    # 2. Add more sample data
    print("\n2. Adding more sample data...")
    
    # Add more parks
    parks = [
        {'id': 'prospect_park', 'type': 'park', 'latitude': 40.6602, 'longitude': -73.9690, 'area': 526, 'population': 0, 'is_public': 1},
        {'id': 'battery_park', 'type': 'park', 'latitude': 40.7032, 'longitude': -74.0170, 'area': 25, 'population': 0, 'is_public': 1},
    ]
    
    # Add more buildings
    buildings = [
        {'id': 'one_world_trade', 'type': 'building', 'latitude': 40.7128, 'longitude': -74.0060, 'area': 150, 'population': 0, 'is_public': 0},
        {'id': 'brooklyn_bridge', 'type': 'building', 'latitude': 40.7061, 'longitude': -73.9969, 'area': 200, 'population': 0, 'is_public': 1},
    ]
    
    # Add more schools
    schools = [
        {'id': 'nyu', 'type': 'school', 'latitude': 40.7295, 'longitude': -73.9969, 'area': 300, 'population': 50000, 'is_public': 1},
        {'id': 'cuny_hunter', 'type': 'school', 'latitude': 40.7685, 'longitude': -73.9646, 'area': 150, 'population': 20000, 'is_public': 1},
    ]
    
    all_nodes = parks + buildings + schools
    for node in all_nodes:
        graph.add_node(node)
    
    # Add spatial relationships
    relationships = [
        # Parks to buildings
        ('prospect_park', 'brooklyn_bridge', 'distance_2km', {'distance': 2000, 'direction': 'northeast'}),
        ('battery_park', 'one_world_trade', 'distance_500m', {'distance': 500, 'direction': 'north'}),
        
        # Schools to parks
        ('nyu', 'central_park', 'distance_3km', {'distance': 3000, 'direction': 'north'}),
        ('cuny_hunter', 'central_park', 'distance_1km', {'distance': 1000, 'direction': 'south'}),
        
        # Buildings to schools
        ('one_world_trade', 'nyu', 'distance_4km', {'distance': 4000, 'direction': 'north'}),
    ]
    
    for source_id, target_id, relation, attrs in relationships:
        source_node = next(n for n in all_nodes if n['id'] == source_id)
        target_node = next(n for n in all_nodes if n['id'] == target_id)
        graph.add_spatial_edge(source_node, target_node, relation, attrs)
    
    print(f"   - Added {len(all_nodes)} new nodes")
    print(f"   - Added {len(relationships)} new relationships")
    
    # 3. Demonstrate spatial queries (without trained model)
    print("\n3. Demonstrating spatial query capabilities...")
    
    # Show graph structure
    print("\n   Graph structure:")
    for node_type in graph.get_types():
        nodes = graph.node_bacward[node_type]
        print(f"   - {node_type}: {len(nodes)} nodes")
        for i, node in enumerate(nodes[:3]):  # Show first 3 nodes
            print(f"     {i+1}. {node.get('id', 'unknown')} at ({node.get('latitude', 0):.4f}, {node.get('longitude', 0):.4f})")
        if len(nodes) > 3:
            print(f"     ... and {len(nodes) - 3} more")
    
    # Show relationships
    print("\n   Spatial relationships:")
    for target_type in graph.edge_list:
        for source_type in graph.edge_list[target_type]:
            for relation_type in graph.edge_list[target_type][source_type]:
                if relation_type != 'self':
                    count = sum(len(edges) for edges in graph.edge_list[target_type][source_type][relation_type].values())
                    print(f"   - {source_type} --[{relation_type}]--> {target_type}: {count} edges")
    
    # 4. Show how to use the query engine (conceptual)
    print("\n4. Spatial Query Engine Interface:")
    print("   The SpatialQueryEngine provides these capabilities:")
    print("   - find_nearest_nodes(query_node, target_type, k=5)")
    print("   - find_adjacent_nodes(query_node, relation_types, max_distance=500)")
    print("   - spatial_classification(query_node)")
    print("   - query(natural_language_query)")
    
    print("\n   Example queries:")
    print("   - 'Find nearest park to Empire State Building'")
    print("   - 'What buildings are adjacent to Central Park?'")
    print("   - 'Classify this location'")
    
    # 5. Training example
    print("\n5. Training the model:")
    print("   To train the spatial HGT model, run:")
    print("   python train_spatial.py --data_dir ./spatial_data --model_dir ./spatial_models")
    print("   ")
    print("   Key parameters:")
    print("   - --sample_depth: How many hops to sample (default: 3)")
    print("   - --sample_width: Nodes per layer per type (default: 50)")
    print("   - --max_distance: Maximum sampling distance in meters (default: 1000)")
    print("   - --n_layers: Number of GNN layers (default: 3)")
    print("   - --n_heads: Number of attention heads (default: 8)")
    
    # 6. Usage with trained model
    print("\n6. Using trained model:")
    print("   Once trained, you can use the model like this:")
    print("   ```python")
    print("   from spatial_query import SpatialQueryEngine")
    print("   ")
    print("   # Load trained model")
    print("   engine = SpatialQueryEngine('./spatial_models/spatial_query.pkl', graph)")
    print("   ")
    print("   # Find nearest park to Empire State Building")
    print("   empire_state = {'id': 'empire_state', 'type': 'building', 'latitude': 40.7484, 'longitude': -73.9857}")
    print("   nearest_parks = engine.find_nearest_nodes(empire_state, 'park', k=3)")
    print("   print(f'Nearest parks: {nearest_parks}')")
    print("   ```")
    
    print("\n=== Example Complete ===")
    print("The spatial HGT implementation is now ready for your spatial queries!")
    print("You can extend this example with your own spatial data and relationships.")

if __name__ == '__main__':
    main()
