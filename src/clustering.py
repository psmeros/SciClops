import re
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder

from utils import analyze_url, read_graph

############################### CONSTANTS ###############################
scilens_dir = str(Path.home()) + '/Dropbox/scilens/cache/diffusion_graph/scilens_3M/'
sciclops_dir = str(Path.home()) + '/Dropbox/sciclops/'

low_memory = True
############################### ######### ###############################

################################ HELPERS ################################
def apriori_clustering():
  te = TransactionEncoder()
  te_ary = te.fit_transform(articles['refs'])

  clusters = apriori(pd.DataFrame(te_ary, columns=te.columns_), min_support=2/len(articles), use_colnames=True, low_memory=low_memory).rename(columns={'itemsets':'refs'})


  def get_articles_for_cluster(cluster_refs):
    return articles[[cluster_refs.issubset(r) for r in articles.refs]].index.tolist()

  clusters['articles'] = clusters.refs.apply(get_articles_for_cluster)
  clusters['length'] = clusters.articles.apply(len)


############################### ######### ###############################


articles = pd.read_csv(scilens_dir + 'article_details_v2.tsv.bz2', sep='\t')#[:1000]
G = read_graph(scilens_dir + 'diffusion_graph_v7.tsv.bz2')
articles['refs'] = articles.url.apply(lambda u: set(G[u]))
articles = articles.set_index('url')

#cleaning
blacklist_refs  = set(open(sciclops_dir + 'blacklist/sources.txt').read().splitlines())
articles['refs'] = articles.refs.apply(lambda r: r - blacklist_refs)

te = TransactionEncoder()
cooc = pd.DataFrame(te.fit_transform(articles['refs']), columns=te.columns_, index=articles.index)

