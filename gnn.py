from torch_geometric.nn import GCNConv, SAGEConv, GATConv
import torch.nn.functional as F
import torch
from torch_geometric.datasets import Planetoid


# PyG install command. At this point, assuming that torch 1.7.0 is installed locally and only the cpu is available, you can view the torch version with the command python -c "import torch; print(torch.__version__)".
# You can see the cuda version by python -c "import torch; print(torch.version.cuda)"
# CC=clang pip install  torch==1.7.0
# CC=clang pip install  torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
# CC=clang pip install  torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
# CC=clang pip install  torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
# CC=clang pip install  torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
# CC=clang pip install  torch-geometric


dataset = Planetoid(root='./tmp/Cora', name='Cora')


class GCN_Net(torch.nn.Module):
    def __init__(self, features, hidden, classes):
        super(GCN_Net, self).__init__()
        self.conv1 = GCNConv(features, hidden)
        self.conv2 = GCNConv(hidden, classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN_Net(dataset.num_node_features, 16, dataset.num_classes).to(device)
data = dataset[0]
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

model.eval()
_, pred = model(data).max(dim=1)
correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print('GCN:', acc)


class GraphSAGE_Net(torch.nn.Module):
    def __init__(self, features, hidden, classes):
        super(GraphSAGE_Net, self).__init__()
        self.sage1 = SAGEConv(features, hidden)
        self.sage2 = SAGEConv(hidden, classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.sage1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.sage2(x, edge_index)

        return F.log_softmax(x, dim=1)


class GAT_Net(torch.nn.Module):
    def __init__(self, features, hidden, classes, heads=1):
        super(GAT_Net, self).__init__()
        self.gat1 = GATConv(features, hidden, heads=heads)
        self.gat2 = GATConv(hidden*heads, classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gat2(x, edge_index)

        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphSAGE_Net(dataset.num_node_features, 16,
                      dataset.num_classes).to(device)
data = dataset[0]
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

model.eval()
_, pred = model(data).max(dim=1)
correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print('GraphSAGE', acc)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GAT_Net(dataset.num_node_features, 16,
                dataset.num_classes, heads=4).to(device)
data = dataset[0]
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

model.eval()
_, pred = model(data).max(dim=1)
correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print('GAT', acc)