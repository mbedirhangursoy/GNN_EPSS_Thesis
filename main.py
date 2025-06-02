from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score, classification_report
from torch_geometric.nn import HeteroConv, GATConv
from torch.nn import Linear
import torch.nn.functional as F
import numpy as np
import torch, csv
from data_related.get_epss_score import *





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


data = torch.load('data_related/my_graph.pt', weights_only=False)

'''epss_scores = []
with open('epss_score2.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        epss_scores.append(float(row[1]))'''

epss_scores = get_logarithmic_epss_score()

model = HeteroGNN(hidden_dim=32, out_dim=1, metadata=data.metadata())


optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


target = torch.tensor(epss_scores, dtype=torch.float)


num_nodes = data['label'].num_nodes
perm = torch.randperm(num_nodes)

train_idx = perm[:int(0.7 * num_nodes)]
test_idx = perm[int(0.7 * num_nodes):]
validation_idx = perm

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[train_idx] = True

test_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask[test_idx] = True

validation_mask = torch.zeros(num_nodes, dtype=torch.bool)
validation_mask[validation_idx] = True

data['label'].train_mask = train_mask
data['label'].test_mask = test_mask
data['label'].validation_mask = validation_mask

def create_classes(value):
    classes = [-10, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    labels = list(range(len(classes) - 1))
    bins = pd.cut([value], bins=classes, labels=labels)[0]

    return bins


def evaluate_logarithmic_multiclass_prediction(mask):
    model.eval()
    out = model(data.x_dict, data.edge_index_dict).squeeze()
    pred = out[mask]
    actual = target[mask]

    actual_classes = []
    pred_classes = []

    for t, o in zip(actual, pred):
        t = t.item()
        o = o.item()
        actual_classes.append(create_classes(t))
        pred_classes.append(create_classes(o))

    accuracy = accuracy_score(actual_classes, pred_classes)
    classification = classification_report(actual_classes, pred_classes)
    confusion_matrix_ = confusion_matrix(actual_classes, pred_classes)
    print(accuracy, classification, confusion_matrix_)
    return accuracy, classification, confusion_matrix_



def evaluate_epss_prediction(mask):
    model.eval()
    out = model(data.x_dict, data.edge_index_dict).squeeze()
    pred = out[mask]
    actual = target[mask]

    hits = []
    all_diffs = []

    for t, o in zip(actual, pred):
        t = t.item()
        o = o.item()
        diff = abs(o - t)
        all_diffs.append(diff)
        if t > 0.8:
            threshold = 0.1
        elif t < 0.1:
            threshold = 0.06 # had to change from 0.05 to 0.06
        else:
            threshold = 0.1 * t  # 10%
        if diff <= threshold:
            hits.append(diff)

    return {
        "hit_rate": np.mean(hits).item(),
        "average_difference": np.mean(all_diffs).item(),
        "median_difference": np.median(all_diffs).item(),
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
    validation_mse = test(data['label'].validation_mask)
    test_mse = test(data['label'].test_mask)
    #print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Validation MSE: {validation_mse:.4f}, Test MSE: {test_mse:.4f}')
    #print(evaluate_epss_prediction(data['label'].test_mask))
    accuracy_log, classification_log, confusion_log = evaluate_logarithmic_multiclass_prediction
    print(f'Accuracy {accuracy_log}')
    print(f'Classification {classification_log}')
    print(f'Confusion Matrix {confusion_log}')

