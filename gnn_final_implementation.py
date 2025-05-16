import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import HeteroConv, GATConv

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

#prepare the data
with open('data_related/h_gnn_output.json') as data_values:
    data_values = json.load(data_values)

    labels = list(data_values.keys())
    attributes = ['basescore', 'baseseverity', 'confidentialityimpact', 'integrityimpact', 'vendor', 'description', 'cwe']


    all_basescores = [[data_values[v]["basescore"]] for v in labels]
    all_baseseverities = [[data_values[v]["baseseverity"]] for v in labels]
    all_confidentialityimpacts = [[data_values[v]["confidentialityimpact"]] for v in labels]
    all_integrityimpacts = [[data_values[v]["integrityimpact"]] for v in labels]
    all_vendors = [[data_values[v]["vendor"]] for v in labels]
    all_descriptions = [data_values[v]["description"] for v in labels]
    all_cwes = [[data_values[v]["cwe"]] for v in labels]

    all_descriptions_together = [' '.join(tokens) for tokens in all_descriptions]


    enc_basescore = OneHotEncoder().fit_transform(all_basescores).toarray()
    enc_baseseverities = OneHotEncoder().fit_transform(all_baseseverities).toarray()
    enc_confidentialityimpact = OneHotEncoder().fit_transform(all_confidentialityimpacts).toarray()
    enc_integrityimpact = OneHotEncoder().fit_transform(all_integrityimpacts).toarray()
    enc_vendor = OneHotEncoder().fit_transform(all_vendors).toarray()
    enc_description = CountVectorizer().fit_transform(all_descriptions_together).toarray()
    enc_cwe = OneHotEncoder().fit_transform(all_cwes).toarray()
    
    
    for i, v in enumerate(labels): #adds the encoded attributes back to the loaded json file
        data_values[v]["basescore"] = enc_cwe[i]
        data_values[v]["baseseverity"] = enc_cwe[i]
        data_values[v]["confidentialityimpact"] = enc_cwe[i]
        data_values[v]["integrityimpact"] = enc_cwe[i]
        data_values[v]["vendor"] = enc_vendor[i]
        data_values[v]["description"] = enc_cwe[i]
        data_values[v]["cwe"] = enc_cwe[i]

    data = HeteroData()

    num_label_nodes = len(labels) # Labeling of node features
    data['label'].x = torch.eye(num_label_nodes)
    label_ids = {val: i for i, val in enumerate(labels)}

    num_attrs = len(attributes) # Attribute node features
    data['attribute'].x = torch.randn(num_attrs, 32)
    attr_ids = {name: i for i, name in enumerate(attributes)}


    edge_index = [[], []]
    edge_attr_list = []
    threshold = 0.8
    description_vector_map = {}  # {desc_idx: vector}
    desc_idx_counter = len(attributes)

    for label_val in labels:
        for attr in attributes:
            label_idx = label_ids[label_val]

            if attr == 'description':
                current_vec = np.array(data_values[label_val][attr])

                matching = False
                for existing_idx, existing_vec in description_vector_map.items():
                    similarity = cosine_similarity([current_vec], [existing_vec])[0][0]
                    if similarity >= threshold:
                        attr_idx = existing_idx
                        matching = True
                        break

                if not matching:
                    attr_idx = desc_idx_counter
                    description_vector_map[attr_idx] = current_vec
                    desc_idx_counter += 1

            else:
                attr_idx = attr_ids[attr]

            edge_index[0].append(label_idx)
            edge_index[1].append(attr_idx)

            attr_val = data_values[label_val][attr]
            if isinstance(attr_val, (np.ndarray, list)):
                attr_val = np.array(attr_val)
            else:
                attr_val = np.array([attr_val]) 

            edge_attr_list.append(attr_val)




    max_len = max(len(e) for e in edge_attr_list)
    edge_attr_array = np.stack([np.pad(e, (0, max_len - len(e)), constant_values=0) for e in edge_attr_list])
    edge_attr = torch.tensor(edge_attr_array, dtype=torch.float)


    edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()
    data['label', 'to', 'attribute'].edge_index = edge_index
    data['label', 'to', 'attribute'].edge_attr = edge_attr
    data['attribute', 'rev_to', 'label'].edge_index = edge_index.flip(0)