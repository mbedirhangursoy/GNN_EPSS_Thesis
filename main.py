from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score, classification_report
from torch_geometric.nn import HeteroConv, GATConv
from torch.nn import Linear
import torch.nn.functional as F
import numpy as np
import torch, csv
from clean_data.helper_functions import *




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


data = torch.load('data_related/my_final_graph_updated.pt', weights_only=False)

epss_scores = []
with open('epss_score_2025_new.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        epss_scores.append(float(row[1]))

### Normalising EPSS Scores
epss_array = np.array(epss_scores, dtype=np.float32) 
mean = epss_array.mean()
std = epss_array.std()
normalized_epss = (epss_array - mean) / std
### Normalising EPSS Scores

model = HeteroGNN(hidden_dim=64, out_dim=1, metadata=data.metadata())


optimizer = torch.optim.Adam(model.parameters(), lr=0.01)



label_node_count = data['label'].num_nodes
#random_index = torch.randperm(label_node_count)
target = torch.tensor(epss_scores, dtype=torch.float)


train_size = int(0.7 * label_node_count)
#train_idx = random_index[:train_size]
#test_idx = random_index[train_size:]
#validation_idx = random_index
train_idx = torch.arange(0, train_size)
test_idx = torch.arange(train_size, label_node_count)
validation_idx = torch.arange(0, label_node_count)

train_mask = torch.zeros(label_node_count, dtype=torch.bool)
test_mask = torch.zeros(label_node_count, dtype=torch.bool)
validation_mask = torch.zeros(label_node_count, dtype=torch.bool)

train_mask[train_idx] = True
test_mask[test_idx] = True
validation_mask[validation_idx] = True

data['label'].train_mask = train_mask
data['label'].test_mask = test_mask
data['label'].validation_mask = validation_mask




def evaluate_logarithmic_multiclass_prediction(mask, start_year):
    model.eval()
    out = model(data.x_dict, data.edge_index_dict).squeeze()
    pred = out[mask]
    actual = target[mask]

    actual_classes = []
    pred_classes = []

    for t, o in zip(actual, pred):
        actual_classes.append(t.item())
        pred_classes.append(o.item())
            
    with open(f'logarithmic_actual_pred_output_{start_year}.csv', 'w') as f: #create a csv for the graph
        writer = csv.writer(f)
        for actual, pred in zip(actual_classes, pred_classes):
            writer.writerow([actual, pred])


    #accuracy = accuracy_score(actual_classes, pred_classes)
    #classification = classification_report(actual_classes, pred_classes)
    #confusion_matrix_ = confusion_matrix(actual_classes, pred_classes)

    #return accuracy, classification, confusion_matrix_

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



def train(epoch):
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict).squeeze()
    pred = out[data['label'].train_mask]
    actual = target[data['label'].train_mask]

    print(pred, actual)
    
    with open(f'all_epochs/train_actual_pred_output_{epoch}.csv', 'w') as f: #create a csv for the graph for train data
        writer = csv.writer(f)
        for predi, actuali in zip(pred, actual):
            writer.writerow([actuali.item(), predi.item()])

    
    #loss = F.mse_loss(pred, actual)
    weights = 1 + 1000 * actual
    loss = torch.mean(weights * (pred - actual) ** 2)

    loss.backward()
    optimizer.step()

    return loss.item()

def test(mask, epoch):
    model.train()

    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict).squeeze()
        pred = out[mask]
        actual = target[mask]
        mse = F.mse_loss(pred, actual).item()


        
        if mask is data['label'].validation_mask:
            with open(f'all_epochs/valid_actual_pred_output_{epoch}.csv', 'w') as f: #create a csv for the graph for validation data
                writer = csv.writer(f)
                for predi, actuali in zip(pred, actual):
                    writer.writerow([actuali.item(), predi.item()])

        elif mask is data['label'].test_mask:
            with open(f'all_epochs/test_actual_pred_output_{epoch}.csv', 'w') as f: #create a csv for the graph for test data
                writer = csv.writer(f)
                for predi, actuali in zip(pred, actual):
                    writer.writerow([actuali.item(), predi.item()])

        return mse

lowest_mse_validation = 1
corressponding_epoch = 0
train_mse_list = []
validation_mse_list = []
test_mse_list = []

for epoch in range(1, 101):
    loss = train(epoch)
    validation_mse = test(data['label'].validation_mask, epoch)
    test_mse = test(data['label'].test_mask, epoch)

    train_mse_list.append(loss)
    validation_mse_list.append(validation_mse)
    test_mse_list.append(test_mse)

    if validation_mse < lowest_mse_validation:
        lowest_mse_validation = validation_mse
        corressponding_epoch = epoch


    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Validation MSE: {validation_mse:.4f}, Test MSE: {test_mse:.4f}')
    #print(evaluate_epss_prediction(data['label'].test_mask))


train_actual, train_pred = load_csv_to_lists('train_actual_pred_output', corressponding_epoch)
val_actual, val_pred = load_csv_to_lists('valid_actual_pred_output', corressponding_epoch)
test_actual, test_pred = load_csv_to_lists('test_actual_pred_output', corressponding_epoch)

plot_actual_vs_predicted(train_actual, train_pred, "Train Set", "train_pred_2025", corressponding_epoch)
plot_actual_vs_predicted(val_actual, val_pred, "Validation Set", "validation_pred_2025", corressponding_epoch)
plot_actual_vs_predicted(test_actual, test_pred, "Test Set", "test_pred_2025", corressponding_epoch)

plot_mse(train_mse_list, validation_mse_list, test_mse_list)