from gnn_final_implementation import epss_scores
from torch_geometric.nn import HeteroConv, GATConv
from torch.nn import Linear
import torch.nn.functional as F
import numpy as np
import torch


print('got epss scores')
#model
class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_dim, out_dim, metadata):
        super().__init__()

        self.conv1 = HeteroConv({
            ('label', 'to', 'attribute'): GATConv((-1, -1), hidden_dim, add_self_loops=False),
            ('attribute', 'rev_to', 'label'): GATConv((-1, -1), hidden_dim, add_self_loops=False)
        }, aggr='sum')

        self.lin = Linear(hidden_dim, out_dim)
        self.metadata = metadata

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)

        x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        out = self.lin(x_dict['label'])
        return out

data = torch.load('data_related/my_graph.pt')

model = HeteroGNN(hidden_dim=32, out_dim=1, metadata=data.metadata())


optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

target = torch.tensor(epss_scores, dtype=torch.float)


num_nodes = data['label'].num_nodes
perm = torch.randperm(num_nodes) #change to num_nodes later if not working

train_idx = perm[:int(0.7 * num_nodes)]
test_idx = perm[int(0.7 * num_nodes):]

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[train_idx] = True

test_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask[test_idx] = True

data['label'].train_mask = train_mask
data['label'].test_mask = test_mask




def evaluate_epss_prediction(mask):
  
    out = model(data.x_dict, data.edge_index_dict).squeeze()
    pred = out[mask]
    actual = target[mask]

    hits = []
    all_diffs = []

    for t, o in zip(actual, pred):
        diff = abs(o - t)
        all_diffs.append(diff.item())
        if t > 0.8:
            threshold = 0.1
        elif t < 0.1:
            threshold = 0.05
        else:
            threshold = 0.1 * t  # 10%

        if diff <= threshold:
            hits.append(diff.item())
    return {
        "hit_rate": np.mean(hits),
        "average_difference": np.mean(all_diffs),
        "median_difference": np.median(all_diffs),
        "total_predictions": len(actual),
        "hits": int(np.sum(hits))
    }

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict).squeeze()
    loss = F.mse_loss(out[data['label'].train_mask], target[data['label'].train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test(mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict).squeeze()
        pred = out[mask]
        actual = target[mask]
        mse = F.mse_loss(pred, actual).item()
        return mse

for epoch in range(1, 101):
    loss = train()
    test_mse = test(data['label'].test_mask)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test MSE: {test_mse:.4f}')
    evaluate_epss_prediction(data['label'].test_mask)  