import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.glob import AvgPooling

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GATLayer, self).__init__()
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    # Modified for edge-weighted attention
    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src["z"], edges.dst["z"]], dim=1)
        a = self.attn_fc(z2)
        e = F.leaky_relu(a)
        
        # Multiply by edge weight
        e = e * edges.data["weight"]
        #print(edges.data["weight"])
        return {"e": e}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {"z": edges.src["z"], "e": edges.data["e"]}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox["e"], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox["z"], dim=1)
        return {"h": h}

    def forward(self, g, h):
        # equation (1)
        z = self.fc(h)
        g.ndata["z"] = z
        # equation (2)
        g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        g.update_all(self.message_func, self.reduce_func)
        return g.ndata.pop("h")

class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, merge="cat"):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for _ in range(num_heads):
            self.heads.append(GATLayer(in_dim, out_dim))
        self.merge = merge

    def forward(self, g, h):
        head_outs = [attn_head(g, h) for attn_head in self.heads]
        if self.merge == "cat":
            # Concatenate the output of each head
            return torch.cat(head_outs, dim=1)
        else:
            # Average the output of each head
            return torch.mean(torch.stack(head_outs), dim=0)


class GAT(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=2, num_classes=2):
        super(GAT, self).__init__()
        # GAT layer with multiple heads
        self.gat1 = MultiHeadGATLayer(in_channels, out_channels, num_heads)
        # Global mean pooling layer
        self.pool = AvgPooling()
        # Fully connected layer for final classification
        self.classifier = nn.Linear(out_channels * num_heads, num_classes)

    def forward(self, g, x):
        # Edge weights should already be stored in g.edata['w']
        # Apply the GAT layer
        x = self.gat1(g, x)
        #x = F.elu(x)
        # Perform global mean pooling
        x = self.pool(g, x)
        print(x.shape)
        # (batch_size, num_heads, out_channels)
        # Reshape to (batch_size, num_heads * out_channels)
        #x = x.view(-1, x.size(1))
        #print(x.shape)
        # Apply the final classifier
        x = self.classifier(x)
        return x