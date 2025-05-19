from gnn_final_implementation import *
from data_related.get_epss_score import get_epss_scores


model = HeteroGNN(hidden_dim=32, out_dim=1, metadata=data.metadata())


optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
print('getting epss scores')
epss_scores = get_epss_scores(list(data_values.keys()), 'epss_score.csv')
data_values2 = data_values
print(len(data_values))
new_epss_scores = []
new_data_values = {}

for score, (key, value) in zip(epss_scores, data_values.items()):
    if score is not None:
        new_epss_scores.append(score)
        new_data_values[key] = value
    else:
        print(f'removed the following EPSS and CVE-ID: {score}, {key}')

epss_scores = new_epss_scores
data_values = new_data_values

print(len(data_values))

target = torch.tensor(epss_scores, dtype=torch.float)

model.train()
for epoch in range(100):
    print('starting')
    optimizer.zero_grad()
    
    out_dict = model(data.x_dict, data.edge_index_dict)
    
    out = out_dict.squeeze()
    
    loss = F.mse_loss(out, target)
    
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")


model.eval()
with torch.no_grad():
    predictions = model(data.x_dict, data.edge_index_dict).squeeze()
    mse = F.mse_loss(predictions, target).item()
    print(f"Mean Squared Error: {mse:.4f}")
