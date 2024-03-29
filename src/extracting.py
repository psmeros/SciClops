import os
import random
import re
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import spacy
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from simpletransformers.classification import ClassificationModel
from simpletransformers.language_modeling import (LanguageModelingArgs, LanguageModelingModel)
from sklearn.metrics import precision_recall_fscore_support

############################### CONSTANTS ###############################
scilens_dir = str(Path.home()) + '/data/scilens/cache/diffusion_graph/scilens_3M/'
sciclops_dir = str(Path.home()) + '/data/sciclops/'
hn_vocabulary = open(sciclops_dir + 'etc/hn_vocabulary/hn_vocabulary.txt').read().splitlines()
action = open(sciclops_dir + 'etc/keywords/action.txt').read().splitlines()
person = open(sciclops_dir + 'etc/keywords/person.txt').read().splitlines()
study = open(sciclops_dir + 'etc/keywords/study.txt').read().splitlines()


np.random.seed(42)

CLAIM_THRESHOLD = 10
LIFT_THRESHOLD = .8
############################### ######### ###############################

################################ HELPERS ################################

#Read diffusion graph
def read_graph(graph_file):
	return nx.from_pandas_edgelist(pd.read_csv(graph_file, sep='\t', header=None), 0, 1, create_using=nx.DiGraph())

#data
nlp = spacy.load('en_core_web_lg')
articles = pd.read_csv(scilens_dir + 'article_details_v3.tsv.bz2', sep='\t').drop_duplicates(subset='url').set_index('url')
tweets = pd.read_csv(scilens_dir + 'tweet_details_v1.tsv.bz2', sep='\t').drop_duplicates(subset='url').set_index('url')
G = read_graph(scilens_dir + 'diffusion_graph_v7.tsv.bz2')


def annotation_sampling(num, max_sents=5):
	sentences = articles[['title', 'full_text']].sample(num)
	sentences = sentences.apply(lambda r: [r['title']] + [re.sub('\n', '', s.text) for _,s in zip(range(max_sents), nlp(r['full_text']).sents) if len(s) >= CLAIM_THRESHOLD and s[0].is_upper], axis=1)
	weights = sentences.apply(lambda l: [len(l) - l.index(s) for s in l])
	
	df = pd.DataFrame([random.choices(sentences[i], weights[i])[0] for i in range(num)], columns=['sentence'])
	df.to_csv(sciclops_dir + 'etc/arguments/validation_set.csv', index=False)

def negative_sampling(num, random_negative=False, max_sents=10):
	if random_negative:
		negative_samples = articles['full_text'].sample(num)
		negative_samples = negative_samples.apply(lambda s: random.choice(list(nlp(s).sents)).text).dropna().to_list()
	else:
		#separate training and testing negative samples
		negative_samples = articles['full_text'].sample(num)
		#split to list of sentences in list of paragraphs
		negative_samples = negative_samples.apply(lambda t: [[re.sub('\n', '', s.text) for _,s in zip(range(max_sents), nlp(p).sents) if len(s) >= CLAIM_THRESHOLD] for p in t.split('\n')[2:-5] if p])
		#compute the probability of a sentence NOT to be a claim
		negative_samples = negative_samples.apply(lambda t: [(s, (t.index(p)/len(t))*(p.index(s)/len(p))) for p in t for s in p])
		#keep the sentence with the max probability
		negative_samples = negative_samples.apply(lambda s: (sorted(s, key=lambda i: i[1])[0][0]) if s else np.nan).dropna().to_list()
			
	negative_samples = pd.DataFrame(negative_samples, columns=['sentence'])
	negative_samples['label'] = 0

	return negative_samples

def process_eval_dataset():
	#round 1
	df = pd.read_csv(sciclops_dir + 'etc/arguments/mturk_results_old.csv')

	df = df[['Input.sentence', 'Input.golden_label', 'Input.type', 'Answer.claim.label', 'LifetimeApprovalRate']]
	df = df.rename(columns={'Input.sentence':'sentence', 'Input.golden_label':'golden_label', 'Input.type':'type', 'Answer.claim.label':'label', 'LifetimeApprovalRate':'approval'})

	#remove spam crowdworkers
	df = df[df.approval.apply(lambda x: int(re.sub(r'\%.*', '', x))) != 0]

	#aggregate results from crowdworkers
	df = pd.DataFrame(df.groupby(['sentence', 'type', 'golden_label'])['label'].apply(lambda x: (lambda c: (c.index[0], 'strong') if c.get(0) - c.get(1, default=0) > 1 else (c.index[0], 'weak') if c.get(0) - c.get(1, default=0) == 1 else np.nan)(x.value_counts())).apply(pd.Series))

	df = df.rename(columns={0:'label', 1: 'agreement'}).reset_index()
	df.label = df.label.map({'Yes':1, 'No':0})
	
	df = df.dropna()[['sentence', 'label', 'agreement']]
	
	df1 = df.copy()

	#round 2
	df = pd.read_csv(sciclops_dir + 'etc/arguments/mturk_results.csv')

	df = df.rename(columns={'Input.sentence':'sentence', 'Answer.False.False':'False', 'Answer.NA.NA':'NA', 'Answer.True.True':'True'})
	df = df[['sentence', 'False', 'NA', 'True']]
	df = df[df['NA']==False]

	df = df.groupby('sentence').idxmax(axis=1).reset_index().rename(columns={ 0:'label'})[['sentence', 'label']]

	df = pd.DataFrame(df.groupby('sentence').apply(lambda x: (lambda c: (c.index[0][1], 'strong') if c.get(0) - c.get(1, default=0) > 1 else (c.index[0][1], 'weak') if c.get(0) - c.get(1, default=0) == 1 else np.nan)(x.value_counts())).apply(pd.Series)).dropna()

	df = df.rename(columns={0:'label', 1: 'agreement'}).reset_index()
	df.label = df.label.map({'True':1, 'False':0})

	df2 = df.copy()

	df = pd.concat([df1, df2])
	df.to_csv(sciclops_dir + 'etc/arguments/mturk_results_full.tsv', sep='\t', index=False)


############################### ######### ###############################

def pretrain_BERT(model_path, use_cuda=False):
	filename = '_df.csv' 
	df = pd.read_csv(sciclops_dir+'etc/million_headlines/abcnews.csv').drop('publish_date', axis=1)
	df.to_csv(filename, index=None, header=False)
	model_args = LanguageModelingArgs()
	model_args.fp16 = False
	model = LanguageModelingModel('bert', model_path, use_cuda=use_cuda, args=model_args)
	model.train_model(filename)
	os.remove(filename)


def evaluate_BERT(model_path, training_set, use_cuda=False, crowd_evaluation=False):

	if crowd_evaluation:
		df = pd.read_csv(training_set, sep='\t')

		model_args = LanguageModelingArgs()
		model_args.fp16 = False
		model = ClassificationModel('bert', model_path, use_cuda=use_cuda, args=model_args)
		model.train_model(df[['sentence', 'label']], args={'overwrite_output_dir':True})

		for crowd_agreement in ['strong', 'weak']:
			df = pd.read_csv(sciclops_dir + 'etc/arguments/mturk_results_full.tsv', sep='\t')
			df = df[(df.agreement == crowd_agreement)]
			df['pred'], _ = model.predict(df['sentence'].to_list())
			result = precision_recall_fscore_support(df['label'], df['pred'], average='binary')
			with open ('results.txt', 'a+') as f: f.write ('Model Path: '+ model_path + '\nTraining set: '+ training_set + '\nCrowd Agreement: '+ crowd_agreement + '\nResult: ' + str(result)+'\n\n\n')

	else:
		df = pd.read_csv(training_set, sep='\t')
		X = df['sentence'].values
		y = df['label'].values
		
		fold = 5
		kf = KFold(n_splits=fold, shuffle=True)
		
		score = 0.0
		for train_index, test_index in kf.split(X):

			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]

			df_train = pd.DataFrame([X_train, y_train]).T.rename(columns={0:'sentence', 1:'label'})
			df_test = pd.DataFrame([X_test, y_test]).T.rename(columns={0:'sentence', 1:'label'})

			model_args = LanguageModelingArgs()
			model_args.fp16 = False
			model = ClassificationModel('bert', model_path, use_cuda=use_cuda, args=model_args)
			model.train_model(df_train, args={'overwrite_output_dir':True})

			df_test['pred'], _ = model.predict(df_test['sentence'].to_list())

			score += accuracy_score(list(df_test['label']), list(df_test['pred']))

		with open ('results.txt', 'a+') as f: f.write ('Model Path: '+ model_path + '\nTraining set: '+ training_set + '\nResult: ' + str(score/fold)+'\n\n\n')


def evaluate_RF(training_set, crowd_evaluation=False):

	if crowd_evaluation:
		df = pd.read_csv(training_set, sep='\t')
		df['vec'] = df['sentence'].apply(lambda s: nlp(s).vector)
		X = np.array(df['vec'].to_list())
		y = np.array(df['label'].to_list())

		model = RandomForestClassifier(random_state=42)
		model.fit(X, y)
	
		for crowd_agreement in ['strong', 'weak']:
			df = pd.read_csv(sciclops_dir + 'etc/arguments/mturk_results_full.tsv', sep='\t')
			df = df[(df.agreement == crowd_agreement)]
			df['vec'] = df['sentence'].apply(lambda s: nlp(s).vector)
			X = np.array(df['vec'].to_list())
			df['pred'] = model.predict(X)
			
			result = precision_recall_fscore_support(df['label'], df['pred'], average='binary')
			with open ('results.txt', 'a+') as f: f.write ('Model Path: Random Forest' + '\nTraining set: '+ training_set + '\nCrowd Agreement: '+ crowd_agreement + '\nResult: ' + str(result)+'\n\n\n')

	else:
		df = pd.read_csv(training_set, sep='\t')
		df['vec'] = df['sentence'].apply(lambda s: list(nlp(s).vector))
		X = np.array(df['vec'].to_list())
		y = np.array(df['label'].to_list())
		
		fold = 5
		kf = KFold(n_splits=fold, shuffle=True)
		
		score = 0.0
		for train_index, test_index in kf.split(X):

			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]

			model = RandomForestClassifier(random_state=42)
			model.fit(X_train, y_train)

			y_pred = model.predict(X_test)

			score += accuracy_score(list(y_test), list(y_pred))

		with open ('results.txt', 'a+') as f: f.write ('Model Path: Random Forest' + '\nTraining set: '+ training_set + '\nResult: ' + str(score/fold)+'\n\n\n')


def use_BERT(model_path, use_cuda=False):
	model_args = LanguageModelingArgs()
	model_args.fp16 = False
	model = ClassificationModel('bert', model_path, use_cuda=use_cuda, args=model_args)

	articles = pd.read_csv(scilens_dir + 'article_details_v3.tsv.bz2', sep='\t')
	titles = articles[['url', 'title']].drop_duplicates(subset='url').rename(columns={'title': 'claim'})
	articles = articles[['url', 'quotes']].drop_duplicates(subset='url')
	articles.quotes = articles.quotes.apply(lambda l: list(map(lambda d: d['quote'], eval(l))))
	articles = articles.explode('quotes').rename(columns={'quotes': 'claim'})
	articles = pd.concat([articles, titles])
	articles = articles[~articles['claim'].isna()]

	articles['label'], _ = model.predict(articles.claim)

	articles = articles[articles.label == 1].drop('label', axis=1)
	articles = articles.groupby('url')['claim'].apply(list).reset_index()
	articles.to_csv(sciclops_dir+'cache/claims_raw.tsv.bz2', sep='\t', index=False)


def baseline(sentence, baseline_type):

	def pattern_search(sentence):
		sentence = nlp(sentence)
		
		entities = [e.text for e in sentence.ents if e.label_ in ['PERSON', 'ORG']]
		verbs = ([w for w in sentence if w.dep_=='ROOT'] or [None])

		for v in verbs:
			if v.text in action:
				for np in v.children:
					if np.dep_ in ['nsubj', 'dobj']:
						claimer = sentence[np.left_edge.i : np.right_edge.i+1].text
						for w in entities:
							if w in claimer:
								return True 
		
			for np in v.children:
				if np.dep_ in ['nsubj', 'dobj']:
					claimer = sentence[np.left_edge.i : np.right_edge.i+1].text
					for w in hn_vocabulary+person+study:
						if w in claimer:
							return True 
	
		return False


	def max_lift(sentence):

		article_url = list(articles[articles['title'].str.find(sentence) != -1].dropna().index) + list(articles[articles['full_text'].str.find(sentence) != -1].dropna().index)

		if not article_url:
			return False

		related_tweets = [tweets.loc[t] for t in G.predecessors(article_url[0]) if t in tweets.index]

		if not related_tweets:
			return False

		overall_popularity =  sum([t['popularity'] for t in related_tweets])
		support = [t['popularity']/overall_popularity for t in related_tweets]

		confidence = [nlp(t['full_text']).similarity(nlp(sentence)) for t in related_tweets]

		max_lift = max([c/s for s,c in zip(support, confidence)])

		if max_lift > LIFT_THRESHOLD:
			return True

		return False

	if baseline_type == 'lift_only':
		return max_lift(sentence)
	elif baseline_type == 'pattern_only':
		return pattern_search(sentence)
	elif baseline_type == 'both_or':
		return max_lift(sentence) or pattern_search(sentence)
	elif baseline_type == 'both_and':
		return max_lift(sentence) and pattern_search(sentence)

def evaluate_baseline(training_set, baseline_type, crowd_evaluation=False):
	if crowd_evaluation:	
		for crowd_agreement in ['strong', 'weak']:
			df = pd.read_csv(sciclops_dir + 'etc/arguments/mturk_results_full.tsv', sep='\t')
			df = df[(df.agreement == crowd_agreement)]
			df['pred'] = df['sentence'].apply(lambda s: baseline(s, baseline_type))
			result = precision_recall_fscore_support(df['label'], df['pred'], average='binary')
			with open ('results.txt', 'a+') as f: f.write ('Model Path: '+ baseline_type + '\nTraining set: '+ training_set + '\nCrowd Agreement: '+ crowd_agreement + '\nResult: ' + str(result)+'\n\n\n')
	else:
		df = pd.read_csv(training_set, sep='\t')
		df['pred'] = df['sentence'].apply(lambda s: baseline(s, baseline_type))
		score = accuracy_score(list(df['label']), list(df['pred']))
		with open ('results.txt', 'a+') as f: f.write ('Model Path: '+ baseline_type + '\nTraining set: '+ training_set + '\nResult: ' + str(score)+'\n\n\n')



if __name__ == "__main__":
	#BERT
	use_cuda = True

	for model_path in ['bert-base-uncased', 'allenai/scibert_scivocab_uncased']:
		pretrain_BERT(model_path=model_path, use_cuda=use_cuda)	
	
	for model_path in ['bert-base-uncased', 'allenai/scibert_scivocab_uncased', sciclops_dir + 'models/NewsBERT', sciclops_dir + 'models/SciNewsBERT']:
		for training_set in [sciclops_dir+'etc/arguments/UKP_IBM.tsv', sciclops_dir+'etc/arguments/UKP_IBM_full.tsv']:
			for crowd_evaluation in [True, False]:
				evaluate_BERT(model_path=model_path, training_set=training_set, use_cuda=use_cuda, crowd_evaluation=crowd_evaluation)

	#RF
	for training_set in [sciclops_dir+'etc/arguments/UKP_IBM.tsv', sciclops_dir+'etc/arguments/UKP_IBM_full.tsv']:
		for crowd_evaluation in [True, False]:
			evaluate_RF(training_set=training_set, crowd_evaluation=crowd_evaluation)

	#Baseline
	for baseline_type in ['pattern_only', 'lift_only']:
		for training_set in [sciclops_dir+'etc/arguments/UKP_IBM.tsv', sciclops_dir+'etc/arguments/UKP_IBM_full.tsv']:
			for crowd_evaluation in [True, False]:
				evaluate_baseline(training_set=training_set, baseline_type=baseline_type, crowd_evaluation=crowd_evaluation)
