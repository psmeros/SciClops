import re
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori


from utils import analyze_url, read_graph

############################### CONSTANTS ###############################
scilens_dir = str(Path.home()) + '/Dropbox/scilens/cache/diffusion_graph/scilens_3M/'
sciclops_dir = str(Path.home()) + '/Dropbox/sciclops/'
############################### ######### ###############################

################################ HELPERS ################################

############################### ######### ###############################

# def outgoing_link_based_clustering():
articles = pd.read_csv(scilens_dir + 'article_details_v2.tsv.bz2', sep='\t')[:1000]
G = read_graph(scilens_dir + 'diffusion_graph_v7.tsv.bz2')
articles['refs'] = articles.url.apply(lambda u: set(G[u]))
articles = articles.set_index('url')

te = TransactionEncoder()
te_ary = te.fit_transform(articles['refs'])

clusters = apriori(pd.DataFrame(te_ary, columns=te.columns_), min_support=2/len(articles), use_colnames=True)

# #all the combinations of references
# def refs_to_clusters(refs, max_combinations=3):
#     return [comb for c in range(max_combinations) for comb in combinations(refs, c+1)]

# articles['clusters'] = articles.apply(lambda a: refs_to_clusters(a.refs), axis=1)

# clusters = articles.clusters.apply(pd.Series) \
#     .merge(articles[['clusters']], right_index = True, left_index = True) \
#     .reset_index() \
#     .drop(['clusters'], axis = 1) \
#     .melt(id_vars = ['url'], value_name = 'clusters') \
#     .drop('variable', axis = 1) \
#     .dropna()

# clusters = clusters.groupby('clusters').url.apply(list).to_frame()

# clusters['length'] = clusters.url.apply(len)


    # # add article to clusters
    # def add_to_clusters(article, refs, clusters, max_combinations=3):
        
    #     for c in range(max_combinations):
    #         for comb in combinations(refs, c+1):

    #             if clusters.loc[clusters.refs == comb].empty:
    #                 clusters = clusters.append({'refs':comb,'articles':[article]}, ignore_index=True)
    #             else:
    #                 clusters.loc[clusters.refs == comb, 'articles'] = clusters.loc[clusters.refs == comb, 'articles'] + [article]


  
    
  #  return clusters