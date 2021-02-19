#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
import os
import glob
import pickle as pkl
import copy
import shutil
import tqdm

from torch.utils.tensorboard import SummaryWriter
from test_tube import Experiment
from rdkit import Chem
from rdkit.Chem import AllChem
from torch import nn
from torch.optim import Adam
from torch.utils.data import Subset

from CoordAE import CoordAE
from MSDScorer import MSDScorer
from KLDLoss import KLDLoss
from data_utils import CODDataset, BlockDataLoader
from test import test


# In[2]:


seed = 0
np.random.seed(0)
torch.manual_seed(0)
np.set_printoptions(precision=5, suppress=True)


# In[3]:


if torch.cuda.is_available() :
    DEVICE = 'cuda'
else :
    DEVICE = 'cpu'


# In[4]:


# handled in args parse for a py script version

n_max = 50
dim_node = 35
dim_edge = 10
nval = 3000
ntst = 3000
hidden_node_dim = 50
dim_f = 100
batch_size = 20
val_num_samples = 5
model_name = 'dl4chem'
savepreddir = 'savepreddir'
use_val = True
mpnn_steps = 5
alignment_type = 'kabsch'
tol = 1e-5
use_X=False
use_R=True
seed=1334
refine_steps=0
refine_mom=0.99
debug = False
useFF = False
w_reg = 1e-5
log_train_steps=100


data_dir = '/home/bb596/rds/hpc-work/dl4chem/'
dataset = 'COD'
COD_molset_50_path = data_dir + 'COD_molset_50.p'  
COD_molvec_50_path = data_dir + 'COD_molvec_50.p'

molvec_fname = data_dir + dataset + '_molvec_'+str(n_max)+'.p'
molset_fname = data_dir + dataset + '_molset_'+str(n_max)+'.p'


# In[5]:


# load data

nodes_fname = data_dir + dataset + '_nodes_'+str(n_max)+'.p'
D1 = pkl.load(open(nodes_fname,'rb'))

masks_fname = data_dir + dataset + '_masks_'+str(n_max)+'.p'
D2 = pkl.load(open(masks_fname,'rb'))

edges_fname = data_dir + dataset + '_edges_'+str(n_max)+'.p'
D3 = pkl.load(open(edges_fname,'rb'))

dist_mats_fname = data_dir + dataset + '_dist_mats_'+str(n_max)+'.p'
D4 = pkl.load(open(dist_mats_fname,'rb'))

positions_fname = data_dir + dataset + '_positions_'+str(n_max)+'.p'
D5 = pkl.load(open(positions_fname,'rb'))


# In[6]:


# TODO : validation set should be a subset of train : to be handled in dataloader class

D1 = D1.todense()
D2 = D2.todense()
D3 = D3.todense()

ntrn = len(D5)-nval-ntst

[molsup, molsmi] = pkl.load(open(molset_fname,'rb'))

D1_trn = D1[:ntrn]
D2_trn = D2[:ntrn]
D3_trn = D3[:ntrn]
D4_trn = D4[:ntrn]
D5_trn = D5[:ntrn]
molsup_trn = molsup[:ntrn]
D1_val = D1[ntrn:ntrn+nval]
D2_val = D2[ntrn:ntrn+nval]
D3_val = D3[ntrn:ntrn+nval]
D4_val = D4[ntrn:ntrn+nval]
D5_val = D5[ntrn:ntrn+nval]
molsup_val = molsup[ntrn:ntrn+nval]
D1_tst = D1[ntrn+nval:ntrn+nval+ntst]
D2_tst = D2[ntrn+nval:ntrn+nval+ntst]
D3_tst = D3[ntrn+nval:ntrn+nval+ntst]
D4_tst = D4[ntrn+nval:ntrn+nval+ntst]
D5_tst = D5[ntrn+nval:ntrn+nval+ntst]
molsup_tst = molsup[ntrn+nval:ntrn+nval+ntst]
print ('::: num train samples is ')
print(D1_trn.shape, D3_trn.shape)

del D1, D2, D3, D4, D5, molsup


# In[7]:


train_dataset = CODDataset(D1_trn[:60000], D2_trn[:60000], D3_trn[:60000], D4_trn[:60000], D5_trn[:60000])
val_dataset = CODDataset(D1_val, D2_val, D3_val, D4_val, D5_val)
test_dataset = CODDataset(D1_tst, D2_tst, D3_tst, D4_tst, D5_tst)


# In[8]:


# train_dataset = Subset(train_dataset, range(100))
# val_dataset = Subset(val_dataset, range(100))
# test_dataset = Subset(test_dataset, range(100))


# In[9]:


train_dataloader = BlockDataLoader(train_dataset, batch_size, block_size=10000)
val_num_samples = 2
val_batch_size = batch_size // val_num_samples
val_dataloader = BlockDataLoader(val_dataset, val_batch_size, block_size=10000, shuffle=False)
test_dataloader = BlockDataLoader(test_dataset, batch_size, block_size=10000, shuffle=False)

# train_dataloader = BlockDataLoader(train_dataset, batch_size)
# val_dataloader = BlockDataLoader(val_dataset, batch_size)
# test_dataloader = BlockDataLoader(test_dataset, batch_size)


# In[ ]:





# In[ ]:





# In[10]:


num_epochs = 100
log_train_steps=1
w_reg=1e-3
exp=None # Experiment

model = CoordAE(n_max, dim_node, dim_edge, hidden_node_dim, dim_f, batch_size,                     mpnn_steps=mpnn_steps, alignment_type=alignment_type, tol=tol,                    use_X=use_X, use_R=use_R, seed=seed,                     refine_steps=refine_steps, refine_mom=refine_mom).to(DEVICE)

model = nn.DataParallel(model)

optimizer = Adam(model.parameters(), lr=3e-4)
kldloss = KLDLoss()
msd_scorer = MSDScorer('default')
    
writer = SummaryWriter()

val_rmsd_means = np.zeros(num_epochs)
val_rmsd_stds = np.zeros(num_epochs)

for epoch in range(num_epochs):

    model.train()
    print('Epoch ' + str(epoch))
    
    print('Training')
    for batch_idx, batch in enumerate(train_dataloader) :
        
        optimizer.zero_grad()
        
        # batch to be created
        nodes, masks, edges, proximity, pos = batch
        nodes = nodes.to(DEVICE)
        masks = masks.to(DEVICE)
        edges = edges.to(DEVICE)
        proximity = proximity.to(DEVICE)
        pos = pos.to(DEVICE)
        masks = masks.unsqueeze(-1) # because dataloader squeezes the mask Tensor for some obscure reason
        
        postZ_mu, postZ_lsgms, priorZ_mu, priorZ_lsgms, X_pred, PX_pred = model(nodes, masks, edges, proximity, pos)
    
        cost_KLDZ = torch.mean(torch.sum(kldloss.loss(masks, postZ_mu, postZ_lsgms,  priorZ_mu, priorZ_lsgms), (1, 2))) # posterior | prior
        cost_KLD0 = torch.mean(torch.sum(kldloss.loss(masks, priorZ_mu, priorZ_lsgms), (1, 2))) # prior | N(0,1)

        cost_X = torch.mean(msd_scorer.score(X_pred, pos, masks))

        cost_op = cost_X + cost_KLDZ + w_reg * cost_KLD0

        # log results
        curr_iter = epoch * len(train_dataloader) + batch_idx

        if curr_iter % log_train_steps == 0:
            writer.add_scalar("train/cost_op", cost_op, curr_iter)
            writer.add_scalar("train/cost_X", cost_X, curr_iter)
            writer.add_scalar("train/cost_KLDZ", cost_KLDZ, curr_iter)
            writer.add_scalar("train/cost_KLD0", cost_KLD0, curr_iter)
        
        cost_op.backward()
        optimizer.step()
    
    exp_dict = {}
    if exp is not None:
        exp_dict['training epoch id'] = epoch
        exp_dict['train_score'] = np.mean(trnscores,0)

    val_mean_rmsd, val_std_rmsd = test(model, val_dataloader, molsup_val, val_num_samples, device=DEVICE)

    val_rmsd_means[epoch] = val_mean_rmsd
    val_rmsd_stds[epoch] = val_std_rmsd

    writer.add_scalar("val/mean_rmsd", val_mean_rmsd, epoch)
    writer.add_scalar("val/min_mean_rmsd", np.min(val_rmsd_means[0:epoch+1]), epoch)
    writer.add_scalar("val/std_rmsd", val_std_rmsd, epoch)
    writer.add_scalar("val/min_std_rmsd", np.min(val_rmsd_stds[0:epoch+1]), epoch)

