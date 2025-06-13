import pandas as pd
import requests
import gzip
import io
import pandas as pd
from datetime import date, timedelta
import csv
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer as wnl
import json, math
import matplotlib.pyplot as plt
import numpy as np
### Everything after here is about modifying the CVE data ###

def df_converter(data): # reads the file and returns the data that is asked for
    with open(data) as file:
        dict_data = json.load(file)
        id = dict_data['cveMetadata']['cveId']
        try:
            metrics = dict_data['containers']['cna']['metrics'][0]['cvssV3_1']
        except KeyError:
            try:
                metrics = dict_data['containers']['cna']['metrics'][0]['cvssV3_0']
            except KeyError:
                try:
                    metrics = dict_data['containers']['adp'][0]['metrics'][0]['cvssV3_1']
                except KeyError:    
                    metrics = {}


        baseseverity = metrics.get('baseSeverity')
        basescore = metrics.get('baseScore')
        confidentialityimpact = metrics.get('confidentialityImpact')
        integrityimpact = metrics.get('integrityImpact')


        try:
            cwe =  dict_data['containers']['cna']['problemTypes'][0]['descriptions'][0]['cweId']
        except KeyError:
            try:
                cwe =  dict_data['containers']['adp'][0]['problemTypes'][0]['descriptions'][0]['cweId']
            except KeyError:
                cwe = None

        try:
            vendor =  dict_data['containers']['cna']['affected'][0]['vendor']
            if vendor == 'n/a':
                vendor = None
        except KeyError:
            vendor = None

        try:
            description =  dict_data['containers']['cna']['descriptions'][0]['value']
        except KeyError:
            description = 'Does not exist'

       
        return id, basescore, baseseverity, confidentialityimpact, integrityimpact, vendor, description, cwe

def tokenize_sentence(sentence):
    if isinstance(sentence, str):
        
        tokens = word_tokenize(sentence)
        filtered_words = [word for word in tokens if word not in stopwords.words('english')]
        lemmatized_word = [wnl().lemmatize(word) for word in filtered_words]
        without_punctuation = [i for i in lemmatized_word if i not in '.,!?']
        
        return without_punctuation

def convert_to_dict(start_year: int, end_year: int):

    cve_dict = {}
    severity_map = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4, None: 0, 'NONE': 0}

    while start_year <= end_year:
        for filesxx in os.listdir(f'data_related/cvelistV5-main/cves/{start_year}/'):
            ds_store_file_location = f'data_related/cvelistV5-main/cves/{start_year}/.DS_Store' #remove all hidden files
            if os.path.isfile(ds_store_file_location):
                os.remove(ds_store_file_location)
            for cve in os.listdir(f'data_related/cvelistV5-main/cves/{start_year}/{filesxx}'):
                print(cve)
                id, basescore, baseseverity, confidentialityimpact, integrityimpact, vendor, description, cwe = df_converter(f'data_related/cvelistV5-main/cves/{start_year}/{filesxx}/{cve}')

                baseseverity = severity_map[baseseverity]
                confidentialityimpact = severity_map[confidentialityimpact]
                integrityimpact = severity_map[integrityimpact]
                description = 'This is empty for now' #tokenize_sentence(description) # tokenise the description

                cve_dict[id] = {'basescore': basescore, 'baseseverity': baseseverity, 'confidentialityimpact': confidentialityimpact, 'integrityimpact': integrityimpact, 'vendor': vendor, 'description': description, 'cwe': cwe}
        
            
        start_year += 1

    return cve_dict


### Everything after is about the EPSS score ###

def request_epss_scores(): #requests epss scores 
    base_url = "https://epss.empiricalsecurity.com/epss_scores-{}.csv.gz"

    epss_dict = {}

    
    current_date = date.today()
    previous_30_day_date = current_date - timedelta(days=30)


    for date_ in (previous_30_day_date + timedelta(n) for n in range(30)):
        date_str = date_.strftime("%Y-%m-%d")
        url = base_url.format(date_str)

        response = requests.get(url)
        response.raise_for_status()


        with gzip.open(io.BytesIO(response.content), 'rt') as f:
            df = pd.read_csv(f, low_memory=False)
            df.columns = df.iloc[0]
            df = df.iloc[1:]
            df = df.reset_index()
            print(date_, df.columns.tolist())
            for _, row in df.iterrows(): 
                cve = row['index'] 
                epss = row['epss']

                if cve not in epss_dict or float(epss) > epss_dict[cve]:
                    epss_dict[cve] = float(epss)

    with open('epss_score_updated.csv', 'w') as f:
        writer = csv.writer(f)
        for cve, score in epss_dict.items():
            writer.writerow([cve, score])
        
    
    #epss_list = [{"cve": cve, "epss": score} for cve, score in epss_dict.items()]

    #return epss_list


def get_epss_score(cve: str): #gets only one score

    #scores = request_epss_scores()
    with open('epss_score_updated.csv') as scores:
        scores = pd.read_csv(scores)
    df = pd.DataFrame.from_dict(scores)
    for data, epss_score in zip(df['cve'], df['epss']):
        if cve == data:
            if epss_score is None:
                return 0
            else:
                return epss_score
                

def get_epss_scores(cve_ids: list[str]): #gets_all_scores
    epss_scores_list = []
    for id in cve_ids:
        epss_scores_list.append(get_epss_score(id))

    return epss_scores_list

'''def get_logarithmic_epss_score(file):
    epss_scores = []
    with open(file) as final_epss_file:
        readCSV = csv.reader(final_epss_file, delimiter=',')
    
        for row in readCSV:
            epss_score = float(row[1])
            if epss_score < 0.1:
                epss_score = round(math.log(epss_score), 1)
            else:
                epss_score = round(epss_score*10, 1)

            epss_scores.append(epss_score)

    return epss_scores

def create_classes(value): #for logarithmic score tracking
    classes = [-float('inf'), -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, float('inf')]
    labels = list(range(len(classes) - 1))
    bins = pd.cut([value], bins=classes, labels=labels)[0]

    return bins'''
          

### Removes from both EPSS list and CVE dictionary ###

def remove_empty_epss_scores(start_year, end_year):
    '''
    This function reads all of the CVE given and removes the ones 
    which do not have an EPSS score. It also saves the graph and the epss score file to be loaded later.
    '''
    print(f"Inside function: removing scores from {start_year} to {end_year}")
    data_values = convert_to_dict(start_year, end_year)

    print('getting epss scores')
    epss_scores = get_epss_scores(list(data_values.keys()))
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

    with open("epss_score_final.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for score, (key, _) in zip(epss_scores, data_values.items()):
            writer.writerow([key, score])

    with open("h_gnn_output_final.json", "w") as outfile:
        json.dump(data_values, outfile)


    return epss_scores, data_values


### Everything below here is about plotting the results ###

def load_csv_to_lists(filename, epoch):
    actual, predicted = [], []
    with open(f'all_epochs/{filename}_{epoch}.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            actual.append(float(row[0]))
            predicted.append(float(row[1]))
    return actual, predicted

def plot_actual_vs_predicted(actual, predicted, title, file_prefix, epoch):
    # full version
    plt.figure(figsize=(10, 6))
    plt.scatter(
        actual, predicted,
        alpha=0.8, s=60, color='dodgerblue',
        edgecolor='black', label='Actual vs. Predicted'
    )


    plt.plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect Prediction')

    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f'{title} - Full Range')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{file_prefix}_{epoch}_full.png', dpi=300)
    plt.show()


    # zoomed
    plt.figure(figsize=(10, 6))
    plt.scatter(
        actual, predicted,
        alpha=0.8, s=60, color='dodgerblue',
        edgecolor='black', label='Actual vs. Predicted'
    )


    plt.plot([0, 0.1], [0, 0.1], 'r--', label='Perfect Prediction')
    plt.xlim(0, 0.1)
    plt.ylim(0, 0.1)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f'{title} - Zoomed (0 to 0.1)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{file_prefix}_{epoch}_zoomed.png', dpi=300)
    plt.show()


def plot_mse(train, validate, test):

    x1 = np.array(train)
    y1 = np.arange(len(train))

    x2 = np.array(validate)
    y2 = np.arange(len(validate))

    x3 = np.array(test)
    y3 = np.arange(len(test))

    plt.plot(x1, y1, label='Train')
    plt.plot(x2, y2, label='Validate')
    plt.plot(x3, y3, label='Test')
    
    plt.xlabel("Mean Squared Error")
    plt.ylabel("Epoch Number")
    plt.title('MSE per Epoch')
    plt.legend()
    plt.grid(True)
    plt.show()