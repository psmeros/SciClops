import re
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

############################### CONSTANTS ###############################
sciclops_dir = str(Path.home()) + '/data/sciclops/'
hn_vocabulary = set(map(str.lower, open(sciclops_dir + 'small_files/hn_vocabulary/hn_vocabulary.txt').read().splitlines()))
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

def popular_clusters():
  NUM_CLUSTERS = 100 
  claims_clusters = pd.read_csv(sciclops_dir + 'cache/claims_clusters.tsv.bz2', sep='\t')
  papers_clusters = pd.read_csv(sciclops_dir + 'cache/papers_clusters.tsv.bz2', sep='\t')
  
  claims_popularity = claims_clusters['popularity']
  papers_popularity = papers_clusters['popularity']
  
  claims_rank = [sum(claims_clusters[str(i)]*claims_popularity) for i in range(NUM_CLUSTERS)]
  claims_rank = [r/sum(claims_rank) for r in claims_rank]
  papers_rank = [sum(papers_clusters[str(i)]*papers_popularity) for i in range(NUM_CLUSTERS)]
  papers_rank = [r/sum(papers_rank) for r in papers_rank]

  rank = np.flip(np.argsort(np.array(claims_rank) + np.array(papers_rank)))

  for c in range(NUM_CLUSTERS):
    print('cluster: ', c)
    claims_centroid = claims_clusters.iloc[claims_clusters[str(c)].argmax()]['claim'].lower()
    print(claims_centroid)
    claims_centroid = set(claims_centroid.split()).intersection(hn_vocabulary)
    
    papers_centroid = papers_clusters.iloc[papers_clusters[str(c)].argmax()]['title'].lower()
    print(papers_centroid)
    papers_centroid = set(papers_centroid.split()).intersection(hn_vocabulary)

    print(claims_centroid.union(papers_centroid))

def clustering_table():
  df = pd.read_csv(sciclops_dir + 'cache/clustering_results.tsv', sep='\t')
  mapping = {'LDA':'LDA', 'GSDMM':'GSDMM', 'GMM':'GMM', 'PCA-GMM':'PCA/GMM', 'KMeans':'K-Means', 'PCA-KMeans':'PCA/K-Means', 'coordinate-align':'GBA-CP', 'compute_P_align_C':'GBA-C', 'compute_C_align_P':'GBA-P', 'coordinate-transform':'GBT-CP', 'compute_P_transform_C':'GBT-C', 'compute_C_transform_P':'GBT-P', 'compute-align-0.1':'AO-Content', 'compute-align-0.5':'AO-Balanced', 'compute-align-0.9':'AO-Graph'}
  df.method = df.method.apply(lambda x: mapping[x])

  df = df.pivot(index='method', columns='clusters', values=['ASW', 'P@3']).reindex(mapping.values()).swaplevel(axis=1).sort_index(axis=1, level=0, sort_remaining=False).applymap(lambda x:'{0:04.1f}%'.format(100 * x))#.round(decimals=3) * 100
  print(df.to_latex())


def sankey_plot():
  CLUSTER = {'num':2, 'label':'stress'}
  #CLUSTER = {'num':5, 'label':'abortion'}
  #CLUSTER = {'num':15, 'label':'tobacco'}
  TOP_K = 10

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
      hoverlabel = dict(font = dict(size = 8))
      ),
      # Add links
      link = dict(
        source =  links['source'],
        target =  links['target'],
        value =  links['popularity'],
        label =  links['text'],
      #color=  links['color'],
      hoverlabel = dict(font = dict(size = 8))
      
    ))])

  fig.update_layout(
      autosize=False,
      width=1000,
      height=1000,
  )

  fig.show()
