from urllib.parse import urlsplit
import re

import networkx as nx
import pandas as pd


#Read diffusion graph
def read_graph(graph_file):
    return nx.from_pandas_edgelist(pd.read_csv(graph_file, sep='\t', header=None), 0, 1, create_using=nx.DiGraph())


#Find the domain and the path of an http url
def analyze_url(url):
    try:
        url=urlsplit(url)
        domain = re.sub(r'^(http(s)?://)?(www\.)?', r'', url.netloc)
        path = '' if domain == '' else url.path
        return domain, path
    except:
        return url, ''
