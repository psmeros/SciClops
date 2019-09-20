import string
from pathlib import Path

import pandas as pd
from nltk.corpus import stopwords
from pandarallel import pandarallel

import pke

############################### CONSTANTS ###############################
scilens_dir = str(Path.home()) + '/data/scilens/cache/diffusion_graph/scilens_3M/'
sciclops_dir = str(Path.home()) + '/data/sciclops/'

############################### ######### ###############################

################################ HELPERS ################################

def keyphrase_extraction(text, n=10):	
	extractor = pke.unsupervised.MultipartiteRank()
	extractor.load_document(input=text)

	pos = {'NOUN', 'PROPN', 'ADJ'}
	stoplist = list(string.punctuation)
	stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
	stoplist += stopwords.words('english')
	extractor.candidate_selection(pos=pos, stoplist=stoplist)
	extractor.candidate_weighting(alpha=1.1, threshold=0.74, method='average')

	keyphrases = extractor.get_n_best(n)
	return keyphrases

############################### ######### ###############################

pandarallel.initialize()

def extract_keyphrases():
	articles = pd.read_csv(scilens_dir + 'article_details_v3.tsv.bz2', sep='\t')
	articles['keyphrases'] = articles.apply(lambda a: keyphrase_extraction(str(a.title) + ' ' + str(a.full_text)), axis=1)
	articles.to_csv(scilens_dir + 'article_details_v4.tsv.bz2', sep='\t', index=False)


ght = list(pd.read_csv(sciclops_dir + 'small_files/global_health_threats/ght.tsv', sep='\t').Keywords)


articles[''] = articles[:3].keyphrases.apply(lambda l: set([kph for sl in [k.split() for (k, _) in eval(l)] for kph in sl if kph in ght]))