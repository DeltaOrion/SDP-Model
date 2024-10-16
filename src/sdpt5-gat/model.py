import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


# Step 5: Define GAT Model
class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAT, self).__init__()
        self.gat1 = GATConv(in_channels, 128, heads=8, dropout=0.6)
        self.gat2 = GATConv(128 * 8, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.gat2(x, edge_index)
        return x  # Return raw logits, sigmoid activation is already handled by the loss function
