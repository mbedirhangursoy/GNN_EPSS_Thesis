import requests
import json



def request_cve_data(): #requests epss scores 
    base_url = 'https://services.nvd.nist.gov/rest/json/cves/2.0'
    response = requests.get(base_url)
    response.raise_for_status()
    data = response.json()

    return data


request_cve_data()