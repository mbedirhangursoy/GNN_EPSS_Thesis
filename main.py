from gnn_final_implementation import *


model = HeteroGNN(hidden_dim=32, out_dim=1, metadata=data.metadata())


optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

target = torch.tensor(epss_scores, dtype=torch.float)


num_nodes = data['label'].num_nodes
perm = torch.randperm(num_label_nodes) #change to num_nodes later if not working

train_idx = perm[:int(0.7 * num_nodes)]
test_idx = perm[int(0.7 * num_nodes):]

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[train_idx] = True

test_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask[test_idx] = True

data['label'].train_mask = train_mask
data['label'].test_mask = test_mask




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

'''model.train()
for epoch in range(100):
    print('starting')
    optimizer.zero_grad()
    
    out_dict = model(data.x_dict, data.edge_index_dict)
    
    out = out_dict.squeeze()
    
    loss = F.mse_loss(out, target)
    
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")'''

