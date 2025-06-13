import torch, json
from torch_geometric.data import HeteroData
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from clean_data.helper_functions import *



remove_empty_epss_scores(2025, 2025)

with open('h_gnn_output_2025_new.json') as data_values:
    print('opened file and starting to create the graph')
    data_values = json.load(data_values)

    labels = list(data_values.keys())
    attributes = ['basescore', 'baseseverity', 'confidentialityimpact', 'integrityimpact', 'vendor', 'cwe']


    all_basescores = [[data_values[v]["basescore"]] for v in labels]
    all_baseseverities = [[data_values[v]["baseseverity"]] for v in labels]
    all_confidentialityimpacts = [[data_values[v]["confidentialityimpact"]] for v in labels]
    all_integrityimpacts = [[data_values[v]["integrityimpact"]] for v in labels]
    all_vendors = [[data_values[v]["vendor"]] for v in labels]
    all_cwes = [[data_values[v]["cwe"]] for v in labels]


    enc_basescore = OneHotEncoder().fit_transform(all_basescores).toarray()
    enc_baseseverities = OneHotEncoder().fit_transform(all_baseseverities).toarray()
    enc_confidentialityimpact = OneHotEncoder().fit_transform(all_confidentialityimpacts).toarray()
    enc_integrityimpact = OneHotEncoder().fit_transform(all_integrityimpacts).toarray()
    enc_vendor = OneHotEncoder().fit_transform(all_vendors).toarray()
    enc_cwe = OneHotEncoder().fit_transform(all_cwes).toarray()


    for i, v in enumerate(labels): #adds the encoded attributes back to the loaded json file
        data_values[v]["basescore"] = enc_basescore[i]
        data_values[v]["baseseverity"] = enc_baseseverities[i]
        data_values[v]["confidentialityimpact"] = enc_confidentialityimpact[i]
        data_values[v]["integrityimpact"] = enc_integrityimpact[i]
        data_values[v]["vendor"] = enc_vendor[i]
        data_values[v]["cwe"] = enc_cwe[i]


    data = HeteroData()

    label_features = []

    for v in labels:
        combined = np.concatenate([
            data_values[v]["basescore"],
            data_values[v]["baseseverity"],
            data_values[v]["confidentialityimpact"],
            data_values[v]["integrityimpact"],
            data_values[v]["vendor"],
            data_values[v]["cwe"]
        ])
        label_features.append(combined)

    data['label'].x = torch.tensor(np.array(label_features), dtype=torch.float)

    
    num_label_nodes = len(labels) # Labeling of node features
    label_ids = {val: i for i, val in enumerate(labels)}
    data['label'].num_nodes = num_label_nodes

    num_attrs = len(attributes) # Attribute node features
    attr_ids = {name: i for i, name in enumerate(attributes)}
    data['attribute'].x = torch.eye(num_attrs)


    edge_index = [[], []]
    edge_attr_list = []


    for label_val in labels:
        for attr in attributes:
            label_idx = label_ids[label_val]

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

    torch.save(data, 'data_related/my_final_graph_updated.pt')
    print("Graph saved successfully.")
