from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import read_graph
import networkx as nx


scilens_dir = str(Path.home()) + '/Dropbox/scilens/'
# hn_vocabulary = open(scilens_dir + 'small_files/hn_vocabulary/hn_vocabulary.txt').read().splitlines()


scilens_dir = scilens_dir + '/cache/diffusion_graph/scilens_3M/'
# #articles = pd.read_csv(scilens_dir + 'article_details_v2.tsv.bz2', sep='\t')
# #G = read_graph(scilens_dir + 'diffusion_graph_v7.tsv.bz2')


papers = pd.read_csv(scilens_dir + 'paper_details_v1.tsv.bz2', sep='\t')

# #def clean_graph(graph_file):

