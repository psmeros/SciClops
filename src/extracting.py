from pathlib import Path

from simpletransformers.classification import ClassificationModel
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import spacy

############################### CONSTANTS ###############################
scilens_dir = str(Path.home()) + '/data/scilens/cache/diffusion_graph/scilens_3M/'
sciclops_dir = str(Path.home()) + '/data/sciclops/'

CLAIM_THRESHOLD = 10
############################### ######### ###############################

################################ HELPERS ################################

#Read diffusion graph
def read_graph(graph_file):
	return nx.from_pandas_edgelist(pd.read_csv(graph_file, sep='\t', header=None), 0, 1, create_using=nx.DiGraph())
############################### ######### ###############################


def train_BERT():
	df = pd.concat([pd.read_csv(sciclops_dir+'small_files/arguments/UKP_IBM.tsv', sep='\t').drop('topic', axis=1), pd.read_csv(sciclops_dir + 'small_files/arguments/scientific.tsv', sep='\t')])
	#train_df, eval_df = train_test_split(df, test_size=0.3, random_state=42)

	# Create a ClassificationModel
	model = ClassificationModel('bert', 'bert-base-uncased', use_cuda=False) # You can set class weights by using the optional weight argument

	# Train the model
	model.train_model(df)

	# Evaluate the model
	#result, model_outputs, wrong_predictions = model.eval_model(eval_df)

def validation_set():	
	nlp = spacy.load('en_core_web_lg')

	articles = pd.read_csv(scilens_dir + 'article_details_v3.tsv.bz2', sep='\t')
	titles = articles[['url', 'title']].drop_duplicates(subset='url')
	tweets = pd.read_csv(scilens_dir + 'tweet_details_v1.tsv.bz2', sep='\t').drop_duplicates(subset='url').set_index('url')

	G = read_graph(scilens_dir + 'diffusion_graph_v7.tsv.bz2')

	titles['prior'] = titles.apply(lambda x: [str(tweets.loc[t]['full_text']).split()[0] == str(x.title).split()[0] for t in G.predecessors(x.url) if t in tweets.index], axis=1).apply(lambda x: sum(x)/len(x) if len(x) else 0)

	#titles['prior'].hist()
	positive_samples = titles[titles.prior == 1].title

	#downsample negative samples
	negative_samples = articles['full_text'].sample(2*len(positive_samples))
	#split to list of sentences in list of paragraphs
	negative_samples = negative_samples.apply(lambda t: [list(nlp(p).sents) for p in t.split('\n')[2:-5] if p])
	#compute the probability of a sentence NOT to be a claim
	negative_samples = negative_samples.apply(lambda t: [(''.join(str(s)), (t.index(p)/len(t))*(p.index(s)/len(p))) for p in t for s in p if len(s) >= CLAIM_THRESHOLD])
	#keep the sentence with the max probability 
	negative_samples = negative_samples.apply(lambda s: max([('',0)]+s, key = lambda i : i[1])[0])

	positive_samples = pd.DataFrame(positive_samples).rename(columns={'title': 'sentence'})
	positive_samples['label'] = 1
	positive_samples

	negative_samples = pd.DataFrame(negative_samples).rename(columns={'full_text': 'sentence'})
	negative_samples = negative_samples[negative_samples.sentence.str.len() != 0][:len(positive_samples)]
	negative_samples['label'] = 0
	negative_samples

	pd.concat([positive_samples, negative_samples]).to_csv(sciclops_dir+'small_files/arguments/scientific.tsv', sep='\t', index=False)


train_BERT()
