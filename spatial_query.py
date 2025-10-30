import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'pyHGT'))

from pyHGT.spatial_data import *
from pyHGT.spatial_model import *
import torch
import numpy as np
from typing import List, Dict, Any, Tuple
import json

class SpatialQueryEngine:
    """
    Spatial query engine for proximity and nearest neighbor queries
    """
    def __init__(self, model_path: str, graph: SpatialGraph, device='cpu'):
        self.graph = graph
        self.device = torch.device(device)
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Get graph metadata
        self.node_types = list(graph.get_types())
        self.num_types = len(self.node_types)
        self.relations = graph.get_meta_graph()
        self.num_relations = len(self.relations)
        
    def _load_model(self, model_path: str):
        """Load trained spatial model"""
        model = SpatialQueryModel(
            in_dim=13,  # Based on spatial feature extractor
            n_hid=400,
            num_types=self.num_types,
            num_relations=self.num_relations,
            n_heads=8,
            n_layers=3,
            n_out=self.num_types,
            dropout=0.2,
            conv_name='spatial_hgt'
        )
        
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        return model
    
    def find_nearest_nodes(self, query_node: Dict, target_type: str, k: int = 5, max_distance: float = 1000.0) -> List[Dict]:
        """
        Find k nearest nodes of target_type to query_node
        
        Args:
            query_node: {'id': str, 'type': str, ...}
            target_type: Type of nodes to find (e.g., 'park', 'school')
            k: Number of nearest nodes to return
            max_distance: Maximum search distance in meters
            
        Returns:
            List of nearest nodes with distances and scores
        """
        # Sample subgraph around query node
        node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = spatial_query_sample(
            0, query_node, self.graph, sample_depth=3, sample_width=50, max_distance=max_distance
        )
        
        # Move to device
        node_feature = node_feature.to(self.device)
        node_type = node_type.to(self.device)
        edge_time = edge_time.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_type = edge_type.to(self.device)
        
        # Get node representations
        with torch.no_grad():
            node_repr = self.model(node_feature, node_type, edge_time, edge_index, edge_type, 'matching')
        
        # Find nodes of target type
        target_type_id = self.node_types.index(target_type) if target_type in self.node_types else -1
        if target_type_id == -1:
            return []
        
        target_mask = (node_type == target_type_id)
        if not target_mask.any():
            return []
        
        target_repr = node_repr[target_mask]
        target_indices = torch.where(target_mask)[0]
        
        # Calculate distances (cosine similarity)
        query_repr = node_repr[0]  # Query node is first
        similarities = torch.cosine_similarity(query_repr.unsqueeze(0), target_repr, dim=1)
        
        # Get top-k most similar
        top_k_indices = torch.topk(similarities, min(k, len(similarities))).indices
        
        results = []
        for idx in top_k_indices:
            node_idx = target_indices[idx].item()
            # Find original node info
            for node_type_name, type_nodes in node_dict.items():
                if node_idx in range(type_nodes[0], type_nodes[0] + len(self.graph.node_bacward[node_type_name])):
                    local_idx = node_idx - type_nodes[0]
                    node_data = self.graph.node_bacward[node_type_name][local_idx]
                    results.append({
                        'node': node_data,
                        'similarity': similarities[idx].item(),
                        'distance': self._calculate_spatial_distance(query_node, node_data)
                    })
                    break
        
        return sorted(results, key=lambda x: x['similarity'], reverse=True)
    
    def find_adjacent_nodes(self, query_node: Dict, relation_types: List[str] = None, max_distance: float = 500.0) -> List[Dict]:
        """
        Find nodes adjacent to query_node based on spatial relationships
        
        Args:
            query_node: {'id': str, 'type': str, ...}
            relation_types: List of relation types to consider (e.g., ['adjacent_to', 'distance_200m'])
            max_distance: Maximum distance for adjacency
            
        Returns:
            List of adjacent nodes
        """
        if relation_types is None:
            relation_types = ['adjacent_to', 'distance_200m', 'distance_400m']
        
        query_type = query_node['type']
        query_id = query_node['id']
        
        adjacent_nodes = []
        
        # Check all possible relationships
        for target_type in self.graph.edge_list:
            if target_type not in self.graph.edge_list:
                continue
                
            for source_type in self.graph.edge_list[target_type]:
                if source_type != query_type:
                    continue
                    
                for relation_type in self.graph.edge_list[target_type][source_type]:
                    if relation_types and relation_type not in relation_types:
                        continue
                        
                    if query_id in self.graph.edge_list[target_type][source_type][relation_type]:
                        for target_id, spatial_attrs in self.graph.edge_list[target_type][source_type][relation_type][query_id].items():
                            distance = spatial_attrs.get('distance', float('inf'))
                            if distance <= max_distance:
                                # Find target node data
                                target_node_data = self.graph.node_bacward[target_type][target_id]
                                adjacent_nodes.append({
                                    'node': target_node_data,
                                    'relation': relation_type,
                                    'distance': distance,
                                    'direction': spatial_attrs.get('direction', 'unknown')
                                })
        
        return sorted(adjacent_nodes, key=lambda x: x['distance'])
    
    def spatial_classification(self, query_node: Dict) -> Dict:
        """
        Classify a node based on its spatial context
        
        Args:
            query_node: {'id': str, 'type': str, ...}
            
        Returns:
            Classification results
        """
        # Sample subgraph
        node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = spatial_query_sample(
            0, query_node, self.graph, sample_depth=2, sample_width=30, max_distance=500
        )
        
        # Move to device
        node_feature = node_feature.to(self.device)
        node_type = node_type.to(self.device)
        edge_time = edge_time.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_type = edge_type.to(self.device)
        
        # Get classification
        with torch.no_grad():
            output = self.model(node_feature, node_type, edge_time, edge_index, edge_type, 'classification')
            probabilities = torch.softmax(output, dim=1)
            
            # Get prediction for query node (first node)
            query_probs = probabilities[0]
            predicted_type_id = torch.argmax(query_probs).item()
            confidence = query_probs[predicted_type_id].item()
            
            # Map to type names
            predicted_type = self.node_types[predicted_type_id] if predicted_type_id < len(self.node_types else 'unknown'
            
            # Get all probabilities
            type_probs = {}
            for i, type_name in enumerate(self.node_types):
                if i < len(query_probs):
                    type_probs[type_name] = query_probs[i].item()
        
        return {
            'predicted_type': predicted_type,
            'confidence': confidence,
            'all_probabilities': type_probs
        }
    
    def _calculate_spatial_distance(self, node1: Dict, node2: Dict) -> float:
        """Calculate spatial distance between two nodes"""
        try:
            from geopy.distance import geodesic
            coord1 = (node1.get('latitude', 0), node1.get('longitude', 0))
            coord2 = (node2.get('latitude', 0), node2.get('longitude', 0))
            return geodesic(coord1, coord2).meters
        except ImportError:
            # Fallback to Euclidean distance if geopy not available
            lat1, lon1 = node1.get('latitude', 0), node1.get('longitude', 0)
            lat2, lon2 = node2.get('latitude', 0), node2.get('longitude', 0)
            return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2) * 111000  # Rough conversion to meters
    
    def query(self, query_text: str, **kwargs) -> Dict:
        """
        Natural language spatial query interface
        
        Examples:
            - "Find nearest park to Empire State Building"
            - "What buildings are adjacent to Central Park?"
            - "Classify this location"
        """
        query_lower = query_text.lower()
        
        if 'nearest' in query_lower and 'park' in query_lower:
            # Extract location from query
            location = self._extract_location_from_query(query_text)
            if location:
                return self.find_nearest_nodes(location, 'park', k=5)
        
        elif 'adjacent' in query_lower or 'near' in query_lower:
            location = self._extract_location_from_query(query_text)
            if location:
                return self.find_adjacent_nodes(location, max_distance=500)
        
        elif 'classify' in query_lower:
            location = self._extract_location_from_query(query_text)
            if location:
                return self.spatial_classification(location)
        
        return {'error': 'Could not parse query', 'query': query_text}
    
    def _extract_location_from_query(self, query_text: str) -> Dict:
        """Extract location information from natural language query"""
        # This is a simple implementation - in practice, you'd use NLP techniques
        query_lower = query_text.lower()
        
        # Simple keyword matching
        if 'empire state' in query_lower:
            return {'id': 'empire_state', 'type': 'building', 'latitude': 40.7484, 'longitude': -73.9857}
        elif 'central park' in query_lower:
            return {'id': 'central_park', 'type': 'park', 'latitude': 40.7829, 'longitude': -73.9654}
        elif 'columbia' in query_lower:
            return {'id': 'columbia_uni', 'type': 'school', 'latitude': 40.8075, 'longitude': -73.9626}
        
        return None


# Example usage functions
def create_sample_spatial_graph():
    """Create sample spatial graph for testing"""
    graph = SpatialGraph()
    
    # Add sample nodes
    nodes = [
        {'id': 'central_park', 'type': 'park', 'latitude': 40.7829, 'longitude': -73.9654, 'area': 843, 'population': 0, 'is_public': 1},
        {'id': 'empire_state', 'type': 'building', 'latitude': 40.7484, 'longitude': -73.9857, 'area': 100, 'population': 0, 'is_public': 0},
        {'id': 'columbia_uni', 'type': 'school', 'latitude': 40.8075, 'longitude': -73.9626, 'area': 200, 'population': 30000, 'is_public': 1},
        {'id': 'bryant_park', 'type': 'park', 'latitude': 40.7536, 'longitude': -73.9832, 'area': 9.6, 'population': 0, 'is_public': 1},
    ]
    
    for node in nodes:
        graph.add_node(node)
    
    # Add spatial relationships
    relationships = [
        ('central_park', 'empire_state', 'distance_400m', {'distance': 400, 'direction': 'south'}),
        ('central_park', 'columbia_uni', 'distance_800m', {'distance': 800, 'direction': 'north'}),
        ('bryant_park', 'empire_state', 'distance_500m', {'distance': 500, 'direction': 'south'}),
    ]
    
    for source_id, target_id, relation, attrs in relationships:
        source_node = next(n for n in nodes if n['id'] == source_id)
        target_node = next(n for n in nodes if n['id'] == target_id)
        graph.add_spatial_edge(source_node, target_node, relation, attrs)
    
    return graph


def main():
    """Example usage of spatial query engine"""
    # Create sample graph
    graph = create_sample_spatial_graph()
    
    # Initialize query engine (you'd need a trained model)
    # engine = SpatialQueryEngine('./spatial_model_save/spatial_query.pkl', graph)
    
    print("Spatial Query Engine created!")
    print("Available query types:")
    print("1. Find nearest park to Empire State Building")
    print("2. Find buildings adjacent to Central Park")
    print("3. Classify a location")
    
    # Example queries (would work with trained model)
    # results = engine.query("Find nearest park to Empire State Building")
    # print(f"Results: {results}")


if __name__ == '__main__':
    main()
