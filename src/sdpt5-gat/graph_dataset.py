import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data

class GraphDataset:
    def __init__(self, input_directory, node_embeddings, device, is_labeled=True):
        self.input_directory = input_directory
        self.node_embeddings = node_embeddings
        self.device = device
        self.is_labeled = is_labeled
        self.nodes_df = pd.read_csv(os.path.join(input_directory, 'nodes.csv'))
        self.edges_df = pd.read_csv(os.path.join(input_directory, 'edges.csv'))

    def build_graphs(self):
        graphs = []
        grouped_nodes = self.nodes_df.groupby('GRAPHID')

        for graph_id, group in grouped_nodes:
            node_indices = group['ID'].values
            node_id_to_index = {node_id: idx for idx, node_id in enumerate(node_indices)}

            num_nodes = len(group)
            x = torch.stack([self.node_embeddings[node_id] for node_id in node_indices]).to(self.device)

            if self.is_labeled:
                y = torch.tensor((group['NUMBER-OF-BUGS'].values >= 1).astype(int), dtype=torch.float).to(self.device)
            else:
                y = None

            graph_edges = self.edges_df[self.edges_df['GRAPHID'] == graph_id]
            source_indices = graph_edges['SOURCE'].map(node_id_to_index).values
            dest_indices = graph_edges['DESTINATION'].map(node_id_to_index).values
            edge_index_np = np.stack([source_indices, dest_indices])
            edge_index = torch.tensor(edge_index_np, dtype=torch.long).to(self.device)

            data = Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)
            data.signature = group['SIGNATURE'].values  # Store signature for output
            graphs.append(data)

        return graphs