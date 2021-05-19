import requests
import os
import pandas as pd
import numpy as np
from pathlib import Path


def fact_check(claim):
    # Define the endpoint (url), payload (sentence to be scored), api-key (api-key is sent as an extra header)
    api_endpoint = "https://idir.uta.edu/claimbuster/api/v2/score/text/"
    request_headers = {"x-api-key": os.getenv('CLAIMBUSTER_KEY')}
    payload = {"input_text": claim}
    # Send the POST request to the API and store the api response
    api_response = requests.post(url=api_endpoint, json=payload, headers=request_headers)
    # Print out the JSON payload the API sent back
    return api_response.json()

df = pd.read_csv(str(Path.home()) + '/data/sciclops/etc/evaluation/raw_claims.csv')
df['fact_check'] = df['Scientific Claim'].apply(fact_check)
df['fact_check'] =  np.digitize(df['fact_check'].apply(lambda fc: fc['results'][0]['score']),[.2, .4, .6, .8, 1]) - 2
df.to_csv(str(Path.home()) + '/data/sciclops/etc/evaluation/claimbuster.csv', index=False)