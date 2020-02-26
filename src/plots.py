import json
import re
import urllib
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
import plotly.graph_objects as go

############################### CONSTANTS ###############################
sciclops_dir = str(Path.home()) + '/data/sciclops/'

CLUSTER = {'num':2, 'label':'stress'}
CLUSTER = {'num':5, 'label':'abortion'}
CLUSTER = {'num':15, 'label':'tobacco'}
TOP_K = 10
############################### ######### ###############################

################################ HELPERS ################################
def analyze_url(url):
	try:
		url=urlsplit(url)
		domain = re.sub(r'^(http(s)?://)?(www\.)?', r'', url.netloc)
		segments = 3 if '.co.' in domain else 2
		domain = '.'.join(domain.split('.')[-segments:])
	except:
		domain = url
	return domain
############################### ######### ###############################

claims_clusters = pd.read_csv(sciclops_dir + 'cache/claims_clusters.tsv.bz2', sep='\t')
papers_clusters = pd.read_csv(sciclops_dir + 'cache/papers_clusters.tsv.bz2', sep='\t')

claims_clusters = claims_clusters.sort_values(by=str(CLUSTER['num']), ascending=False)
claims_clusters['domain'] = claims_clusters.url.apply(analyze_url)
claims_clusters = claims_clusters.drop_duplicates(subset='domain', keep='first')[:TOP_K]
claims_clusters = claims_clusters[['claim', 'popularity', 'domain']].rename(columns={'claim':'text'})

papers_clusters = papers_clusters.sort_values(by=str(CLUSTER['num']), ascending=False)
papers_clusters['domain'] = papers_clusters.url.apply(analyze_url)
papers_clusters = papers_clusters[~papers_clusters['domain'].isin(['commondreams.org'])]
papers_clusters = papers_clusters[:TOP_K]
papers_clusters = papers_clusters[['title', 'popularity', 'domain']].rename(columns={'title':'text'})

claims_clusters['label'] = claims_clusters['domain']# + ': ' + claims_clusters['claim']
claims_clusters['color'] = 'red'
papers_clusters['label'] = papers_clusters['domain']# + ': ' + papers_clusters['title']
papers_clusters['color'] = 'blue'

claims_clusters['popularity'] /= max(claims_clusters['popularity'])
papers_clusters['popularity'] /= max(papers_clusters['popularity'])

links = pd.concat([claims_clusters[['popularity', 'text', 'color']], papers_clusters[['popularity', 'text', 'color']]], ignore_index=True)
links['source'] = pd.Series(list(range(TOP_K)) + TOP_K*[2*TOP_K])
links['target'] = pd.Series(TOP_K*[2*TOP_K] + list(range(TOP_K, 2*TOP_K)))

nodes = pd.concat([claims_clusters[['label', 'color']], papers_clusters[['label', 'color']], pd.DataFrame([{'label':CLUSTER['label'], 'color':'purple'}])], ignore_index=True)

fig = go.Figure(data=[go.Sankey(
    textfont = dict(size = 8),
    # Define nodes
    node = dict(
      pad = 15,
      thickness = 15,
      line = dict(color = "black", width = 0.5),
      label =  nodes['label'],
      color =  nodes['color'],
	  hoverlabel = dict(font = dict(size = 6))
    ),
    # Add links
    link = dict(
      source =  links['source'],
      target =  links['target'],
      value =  links['popularity'],
      label =  links['text'],
	  #color=  links['color'],
	  hoverlabel = dict(font = dict(size = 6))
  ))])

fig.show()
