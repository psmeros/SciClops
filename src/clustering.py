import re
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import spacy
import torch
import torch.nn as nn
from sklearn.preprocessing import MultiLabelBinarizer
from torch.autograd import Variable
from torch.nn import init
from torch.nn.functional import gumbel_softmax
from torch.optim import SGD

############################### CONSTANTS ###############################
scilens_dir = str(Path.home()) + '/Dropbox/scilens/cache/diffusion_graph/scilens_3M/'
sciclops_dir = str(Path.home()) + '/Dropbox/sciclops/'

nlp = spacy.load("en_core_sci_md")

# Hyper Parameters
num_epochs = 60
learning_rate = 1.e-5
weight_decay = 0.0
gumbel_tau = 1

############################### ######### ###############################

################################ HELPERS ################################

#Read diffusion graph
def read_graph(graph_file):
    return nx.from_pandas_edgelist(pd.read_csv(graph_file, sep='\t', header=None), 0, 1, create_using=nx.DiGraph())


def data_preparation(limit=100):

  articles = pd.read_csv(scilens_dir + 'article_details_v2.tsv.bz2', sep='\t')[:limit]
  papers = pd.read_csv(scilens_dir + 'paper_details_v1.tsv.bz2', sep='\t')
  G = read_graph(scilens_dir + 'diffusion_graph_v7.tsv.bz2')
  articles['refs'] = articles.url.apply(lambda u: set(G[u]))
  articles = articles.set_index('url')

  #cleaning
  blacklist_refs  = set(open(sciclops_dir + 'blacklist/sources.txt').read().splitlines())
  articles['refs'] = articles.refs.apply(lambda r: (r - blacklist_refs).intersection(set(papers.url.to_list())))
  mlb = MultiLabelBinarizer()
  cooc = pd.DataFrame(mlb.fit_transform(articles.refs), columns=mlb.classes_, index=articles.index)
  papers = papers[papers.url.isin(list(cooc.columns))]

  articles_vec = articles.apply(lambda x: nlp(x['title'] + ' ' + x['full_text']).vector , axis=1)
  papers_vec = papers.apply(lambda x: nlp(x['title'] + ' ' + x['full_text']).vector , axis=1)

  
  articles_vec = torch.Tensor(np.matrix(articles_vec.values.tolist()))
  papers_vec = torch.Tensor(np.matrix(papers_vec.values.tolist()))
  cooc = torch.Tensor(cooc.values)

  return articles_vec, papers_vec, cooc


############################### ######### ###############################

class ClusterNet(nn.Module):
    def __init__(self, num_articles, num_papers, embeddings_dim, num_clusters):
        super(ClusterNet, self).__init__()
        self.num_articles = num_articles
        self.num_papers = num_papers
        self.embeddings_dim = embeddings_dim
        self.num_clusters = num_clusters
        
        self.linear_a = nn.Linear(embeddings_dim, num_clusters)
        self.linear_p = nn.Linear(embeddings_dim, num_clusters)

        # #loss functtion
        # self.lam = config.lam #balance clusters
        # self.loss_fact = config.loss_fact

  
    def forward(self, articles, papers, cooc):
        A = self.linear_a(articles)
        P = self.linear_p(papers)

        A = gumbel_softmax(A, tau=gumbel_tau, hard=True)
        P = gumbel_softmax(P, tau=gumbel_tau, hard=True)
        return A.t() @ cooc @ P
        # self.user_assign_tensor, _ = gumbel_sinkhorn(self.user_params, temp=self.temp, noise_factor=self.noise_factor, n_iters=self.num_iters, n_samples=self.gumbel_samples, do_reshape=self.gumbel_samples==1)  
        # self.item_assign_tensor, _ = gumbel_sinkhorn(self.item_params, temp=self.temp, noise_factor=self.noise_factor, n_iters=self.num_iters, n_samples=self.gumbel_samples, do_reshape=self.gumbel_samples==1)  

        # if use_argmax and use_argmax:
        #     Auz = torch.zeros_like(self.user_assign_tensor).squeeze()
        #     veclen = len(self.user_assign_tensor.squeeze().argmax(1))
        #     Auz[range(veclen),self.user_assign_tensor.squeeze().argmax(1)] = 1.0 
        #     self.user_assign_tensor = Auz

        #     Aiz = torch.zeros_like(self.item_assign_tensor).squeeze()
        #     veclen = len(self.item_assign_tensor.squeeze().argmax(1))
        #     Aiz[range(veclen),self.item_assign_tensor.squeeze().argmax(1)] = 1.0 
        #     self.item_assign_tensor = Aiz

        # #i_embds = torch.index_select(self.item_embs, 0, item_ids)

        # u_assigns = torch.index_select(self.user_assign_tensor.squeeze(), 0, user_ids)
        # i_assigns = torch.index_select(self.item_assign_tensor.squeeze(), 0, item_ids)

        # b_embds = torch.index_select(self.item_bias, 0, item_ids)

        # arch_u = (u_assigns @ self.arch_u_embs)
        # arch_i = (i_assigns @ self.arch_i_embs)

        # pred_mat = (arch_u * arch_i).sum(1) + b_embds.squeeze()

        # return pred_mat

    def loss(self, adj, uis, iis, balance_node=False):
        self.user_assign_tensor, _ = gumbel_sinkhorn(self.user_params, temp=self.temp, noise_factor=self.noise_factor, n_iters=self.num_iters, n_samples=self.gumbel_samples, do_reshape=self.gumbel_samples==1)  
        self.item_assign_tensor, _ = gumbel_sinkhorn(self.item_params, temp=self.temp, noise_factor=self.noise_factor, n_iters=self.num_iters, n_samples=self.gumbel_samples, do_reshape=self.gumbel_samples==1)  

        adj = adj.view(1, self.num_users, self.num_items)
        if self.gumbel_samples!=1:
            adj = adj.repeat(self.gumbel_samples, 1, 1)
        
        adj = torch.transpose(self.user_assign_tensor, 1, 2) @ adj @ self.item_assign_tensor 
        self.ypred = adj.view(self.gumbel_samples, self.num_parts, self.num_parts)

        loss = 0
        ncut_loss = torch.sum(torch.sum(torch.tril(self.ypred, diagonal=-1), dim=1), dim=1) + torch.sum(torch.sum(torch.triu(self.ypred, diagonal=1), dim=1), dim=1)

        if balance_node:
            balance_loss_user = torch.sum((torch.sum(self.user_assign_tensor, dim=1) - self.num_users//self.num_parts)**2)
            balance_loss_item = torch.sum((torch.sum(self.item_assign_tensor, dim=1) - self.num_items//self.num_parts)**2)
            balance_loss =  balance_loss_user + balance_loss_item
        else:
            balance_loss = torch.sum((torch.diagonal(self.ypred, 0, dim1=-2, dim2=-1) - torch.sum(torch.diagonal(self.ypred, 0, dim1=-2, dim2=-1), dim=1).unsqueeze(dim=1).repeat(1, self.num_parts)/self.num_parts)**2)
        
        diff_loss = torch.mean(self.lam*ncut_loss + (1-self.lam)*balance_loss)
        
        #mse_loss = self.mse(adj*mask_tr, pred_mat*mask_tr)

        loss += self.loss_fact * diff_loss #* mse_loss

        return loss


articles_vec, papers_vec, cooc = data_preparation()

#Model training
model = ClusterNet(num_articles=len(articles_vec), num_papers=len(papers_vec), embeddings_dim=200, num_clusters=50)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay) 

for epoch in range(num_epochs):    
    optimizer.zero_grad()
    D = model(articles_vec, papers_vec, cooc)

    optimizer.step()
