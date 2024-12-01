import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.glob import AvgPooling
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.glob import AvgPooling
from dgl.nn.pytorch.conv import GraphConv

class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=2, num_classes=2):
        super(GCN, self).__init__()
        total_out_channels = out_channels * num_heads
        self.conv1 = GraphConv(in_channels, total_out_channels)
        self.pool = AvgPooling()
        self.classifier = nn.Linear(total_out_channels, num_classes)

    def forward(self, g, x):
        x = self.conv1(g, x)
        x = F.relu(x)
        x = self.pool(g, x)
        x = self.classifier(x)
        return x
