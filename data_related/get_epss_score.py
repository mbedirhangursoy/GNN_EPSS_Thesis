import pandas as pd
import requests
import gzip
import io
import pandas as pd
from datetime import date, timedelta
import csv



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
            df = pd.read_csv(f)
            df.columns = df.iloc[0]
            df = df.iloc[1:]
            df = df.reset_index()
            print('hello')
            print(df.columns.tolist())
            for _, row in df.iterrows(): 
                cve = row['index'] 
                epss = row['epss']

                if cve not in epss_dict or float(epss) > epss_dict[cve]:
                    epss_dict[cve] = float(epss)
    
    epss_list = [{"cve": cve, "epss": score} for cve, score in epss_dict.items()]
    keys = epss_list[0].keys()

    with open("epss_score.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in epss_list:
            w.writerow(row)

    

def get_epss_score(cve: str, file_location: str): #gets only one score
    with open(file_location) as scores:
        df = pd.read_csv(scores)
        for data, epss_score in zip(df['cve'], df['epss']):
            if cve == data:
                if epss_score is None:
                    return 0
                else:
                    return epss_score
            

def get_epss_scores(cve_ids: list[str], file_location: str): #gets_all_scores
    epss_scores_list = []
    for id in cve_ids:
        epss_scores_list.append(get_epss_score(id, file_location))

    return epss_scores_list

