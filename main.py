from gnn_final_implementation import *
from data_related.get_epss_score import get_epss_scores


model = HeteroGNN(hidden_dim=32, out_dim=1, metadata=data.metadata())


optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
print('getting epss scores')
epss_scores = get_epss_scores(list(data_values.keys()), 'epss_score.csv')

for each_score, values in zip(epss_scores, data_values.copy()): #remove data_values which do not have an EPSS score 
    if each_score is None:
        epss_scores.remove(each_score)
        del data_values[values]

        print(f'removed the following EPSS and CVE-ID: {each_score}, {values}')

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
    predictions = model(data.x_dict, data.edge_index_dict)['label'].squeeze()
    mse = F.mse_loss(predictions, target).item()
    print(f"Mean Squared Error: {mse:.4f}")
