A Heterogeneous GNN Algorithm for Prediction of Static EPSS Scores

About
The EPSS scoring process is widely used to predict the exploitability of software flaws, 
but its utilisation of commercial data and black- box analysis limits transparency, reproducibility, 
and research activ- ity. This paper explores the possibility of creating an open-source alternative by 
predicting exploitability scores from just public data, hoping to match EPSS’s level of accuracy. In direction 
of this ob- jective, we constructed a heterogeneous Graph Neural Network that represents CVEs and associated 
attributes as graph nodes in an organized graph. The model is trained on data from the NVD and EPSS datasets 
to predict exploitability via supervised regression, with experiments conducted across various architectural 
configurations to avoid underfitting and overfitting with dropout and dimensionality regulation.


Important files to consider:
- helper_functions.py
- gnn_final_implementation.py
- main.py

helper_functions: This extracts important information needed from our CVE data and saves them as a JSON file. It also gets the EPSS scores from an API and removes any CVE's which do not have an EPSS score.
gnn_final_implementation: creates vectors of the attributes and the graph to be used for GNN
main: loads the saved graph, splits train, validationa and test data and runs experiments with our data.
