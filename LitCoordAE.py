import torch
import copy
import numpy as np

from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import Adam
from MSDScorer import MSDScorer
from KLDLoss import KLDLoss
from argparse import ArgumentParser

from rdkit import Chem
from rdkit.Chem import AllChem

# Naming of variables (ie original nodes embed vs nodes embed), keep the same terminology

# 3D Coordinates autoencoder model
class LitCoordAE(LightningModule):

    def __init__(self, hparams, *args, **kwargs):
        
        super(LitCoordAE, self).__init__()
        
        self.hparams = hparams # store all the hyperparameters of the model. 
        # See argument parser (model and in training script to find the parameters and a short description)
        
        self.kldloss = KLDLoss()
        self.msd_scorer = MSDScorer(alignment_type=hparams.alignment_type)
        
        # Difference from original model : self.G not included ; prior_T not included
        
        self.embed_nodes = EmbedNode(hparams.batch_size, hparams.n_max, hparams.dim_node, hparams.dim_h)
        
        # Prior Z
        self.edge_nn_prior_z = EmbedEdge(hparams.batch_size, hparams.n_max, hparams.dim_edge + 1, hparams.dim_h)
        self.mpnn_prior_z = MPNN(hparams.batch_size, hparams.n_max, hparams.dim_h, hparams.mpnn_steps)
        self.latent_nn_prior_z = LatentNN(hparams.batch_size, hparams.n_max, hparams.dim_h, hparams.dim_f, 2*hparams.dim_h)
        
        # Post Z
        if self.hparams.use_R :
            self.edge_nn_post_z = EmbedEdge(hparams.batch_size, hparams.n_max, hparams.dim_edge + 2, hparams.dim_h)
        else :
            self.edge_nn_post_z = EmbedEdge(hparams.batch_size, hparams.n_max, hparams.dim_edge + 1, hparams.dim_h)
            
        if self.hparams.use_X:
            self.embed_nodes_pos = EmbedNode(hparams.batch_size, hparams.n_max, hparams.dim_node + 3, hparams.dim_h)
            
        self.mpnn_post_z = MPNN(hparams.batch_size, hparams.n_max, hparams.dim_h, hparams.mpnn_steps)
        self.latent_nn_post_z = LatentNN(hparams.batch_size, hparams.n_max, hparams.dim_h, hparams.dim_f, 2*hparams.dim_h)
        
        # Post X
        self.edge_nn_post_x = EmbedEdge(hparams.batch_size, hparams.n_max, hparams.dim_edge + 1, hparams.dim_h)
        self.mpnn_post_x = MPNN(hparams.batch_size, hparams.n_max, hparams.dim_h, hparams.mpnn_steps)
        self.latent_nn_post_x = LatentNN(hparams.batch_size, hparams.n_max, hparams.dim_h, hparams.dim_f, 3)
        
#         # Pred X
#         self.edge_nn_pred_x = EmbedEdge(batch_size, n_max, dim_edge + 1, dim_h)
#         self.mpnn_pred_x = MPNN(batch_size, n_max, dim_h, mpnn_steps)
#         self.latent_nn_pred_x = LatentNN(batch_size, n_max, dim_h, dim_f, 3)
        
    def forward(self, nodes, masks, edges, proximity, pos) :

        """ Args :
                nodes : Tensor(batch_size, n_max, node_feature_size)
                masks : Tensor(batch_size, n_max, 1)
                edges : Tensor(batch_size, n_max, n_max, dim_edge)
                pos : Tensor(batch_size, n_max, 3)
                proximity : Tensor(batch_size, n_max, n_max) is actually a distance matrix
            Returns :

        """
        
        nodes_embed = self.embed_nodes(nodes, masks) # (batch_size, n_max, dim_h)

        # Isn't there a better way to add n_atom in edge features ?
        n_atom = masks.permute(0, 2, 1).sum(2) # (batch_size, 1)
        tiled_n_atom = n_atom.view(-1, 1, 1, 1).repeat(1, self.hparams.n_max, self.hparams.n_max, 1) # (batch_size, n_max, n_max, 1)
        edge_2 = torch.cat([edges, tiled_n_atom], 3) # (batch_size, n_max, nmax, dim_edge + 1)


        # p(Z|G) -- prior of Z

        priorZ_edge_wgt = self.edge_nn_prior_z(edge_2) #[batch_size, n_max, n_max, dim_h, dim_h]
        priorZ_hidden = self.mpnn_prior_z(priorZ_edge_wgt, nodes_embed, masks) # (batch_size, n_max, dim_h), nodes_embed like
        priorZ_out = self.latent_nn_prior_z(priorZ_hidden, nodes_embed, masks) # (batch_size, n_max, 2*dim_h)

        priorZ_mu, priorZ_lsgms = priorZ_out.split([self.hparams.dim_h, self.hparams.dim_h], 2)
        priorZ_sample = self._draw_sample(priorZ_mu, priorZ_lsgms, masks) # (batch_size, n_max, dim_h)


        # q(Z|R(X),G) -- posterior of Z

        if self.hparams.use_R:
            proximity_view = proximity.view(-1, self.hparams.n_max, self.hparams.n_max, 1)
            edge_cat = torch.cat([edge_2, proximity_view], 3) #[batch_size, n_max, n_max, dim_edge + 2]
            postZ_edge_wgt = self.edge_nn_post_z(edge_cat) #[batch_size, n_max, n_max, dim_h, dim_h]
        else:
            postZ_edge_wgt = self.edge_nn_post_z(edge_2) 

        if self.hparams.use_X:
            nodes_pos = torch.cat([nodes, pos], 2) # (batch_size, n_max, dim_node + 3)
            nodes_pos_embed = self.embed_nodes_pos(nodes_pos, masks)
            postZ_hidden = self.mpnn_post_z(postZ_edge_wgt, nodes_pos_embed, masks)
        else:
            postZ_hidden = self.mpnn_post_z(postZ_edge_wgt, nodes_embed, masks)

        postZ_out = self.latent_nn_prior_z(postZ_hidden, nodes_embed, masks)

        postZ_mu, postZ_lsgms = postZ_out.split([self.hparams.dim_h, self.hparams.dim_h], 2)
        postZ_sample = self._draw_sample(postZ_mu, postZ_lsgms, masks)


        # p(X|Z,G) -- posterior of X

        X_edge_wgt = self.edge_nn_post_x(edge_2) #[batch_size, n_max, n_max, dim_h, dim_h]
        X_hidden = self.mpnn_post_x(X_edge_wgt, postZ_sample + nodes_embed, masks)
        X_pred = self.latent_nn_post_x(X_hidden, nodes_embed, masks)


        # p(X|Z,G) -- posterior of X without sampling from latent space
        # used for iterative refinement of predictions ; det stands for deterministic

#         X_edge_wgt_det = self.edge_nn_post_x_det(edge_2) #[batch_size, n_max, n_max, dim_h, dim_h]
#         X_hidden_det = self.mpnn_post_x_det(X_edge_wgt_det, postZ_mu + nodes_embed, masks)
#         X_pred_det = self.latent_nn_post_x_det(X_hidden_det, nodes_embed, masks)


        # Prediction of X with p(Z|G) in the test phase

        PX_edge_wgt = self.edge_nn_post_x(edge_2) #[batch_size, n_max, n_max, dim_h, dim_h]
        PX_hidden = self.mpnn_post_x(PX_edge_wgt, priorZ_sample + nodes_embed, masks)
        PX_pred = self.latent_nn_post_x(PX_hidden, nodes_embed, masks)

        return postZ_mu, postZ_lsgms, priorZ_mu, priorZ_lsgms, X_pred, PX_pred

    
    def training_step(self, batch, batch_idx):
        
        tensors, mols = batch
        nodes, masks, edges, proximity, pos = tensors
        #masks = masks.unsqueeze(-1)
        
        postZ_mu, postZ_lsgms, priorZ_mu, priorZ_lsgms, X_pred, PX_pred = self(nodes, masks, edges, proximity, pos)
        
        cost_KLDZ = torch.mean(torch.sum(self.kldloss.loss(masks, postZ_mu, postZ_lsgms,  priorZ_mu, priorZ_lsgms), (1, 2))) # posterior | prior
        cost_KLD0 = torch.mean(torch.sum(self.kldloss.loss(masks, priorZ_mu, priorZ_lsgms), (1, 2))) # prior | N(0,1)
        cost_X = torch.mean(self.msd_scorer.score(X_pred, pos, masks))
        cost_op = cost_X + cost_KLDZ + self.hparams.w_reg * cost_KLD0

        self.log("train/cost_op", cost_op)
        self.log("train/cost_X", cost_X)
        self.log("train/cost_KLDZ", cost_KLDZ)
        self.log("train/cost_KLD0", cost_KLD0)
        return cost_op
    
    
    def validation_step(self, batch, batch_idx):
        
        tensors, mols = batch
        nodes, masks, edges, proximity, pos = tensors
        #masks = masks.unsqueeze(-1)

        # repeat because we want val_batch_size (molecule) * val_num_samples (conformer per molecule)
        nodes = torch.repeat_interleave(nodes, self.hparams.val_num_samples, dim=0)
        masks = torch.repeat_interleave(masks, self.hparams.val_num_samples, dim=0)
        edges = torch.repeat_interleave(edges, self.hparams.val_num_samples, dim=0)
        proximity = torch.repeat_interleave(proximity, self.hparams.val_num_samples, dim=0)

        _, _, _, _, _, PX_pred = self(nodes, masks, edges, proximity, pos)

        X_pred = PX_pred
        for r in range(self.hparams.refine_steps):
            if use_X:
                pos = X_pred
            if use_R:
                proximity = pos_to_proximity(X_pred, masks)
            _, _, _, _, last_X_pred, _ = self(nodes, masks, edges, proximity, pos)
            X_pred = self.hparams.refine_mom * X_pred + (1-self.hparams.refine_mom) * last_X_pred

        rmsds=[]
        for j in range(X_pred.shape[0]):
            ms_v_index = int(j / self.hparams.val_num_samples)
            rmsds.append(self.getRMSD(mols[ms_v_index], X_pred[j]))

        rmsds = np.array(rmsds)

        self.log("val/mean_rmsd", rmsds.mean())
        self.log("val/std_rmsd", rmsds.std())
        return rmsds.mean()
        
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3) # original is lr=3e-4
    
    
    def pos_to_proximity(self, pos, mask):
        """ Args
                pos : Tensor(batch_size, n_max, 3)
                mask : Tensor(batch_size, n_max, 1)

            Returns :
                proximity : Tensor(batch_size, n_max, nmax)
        """

        pos_1 = pos.unsqueeze(2)
        pos_2 = pos.unsqueeze(1)

        pos_sub = torch.sub(pos_1, pos_2) #[batch_size, n_max, nmax, 3]
        proximity = torch.square(pos_sub)
        proximity = torch.sum(proximity, 3) #[batch_size, n_max, nmax]
        proximity = torch.sqrt(proximity + 1e-5)

        #proximity_view = torch.view(self.batch_size, self.n_max, self.n_max) I don't understand the rationale
        proximity = torch.mul(proximity, mask)
        proximity = torch.mul(proximity, mask.permute(0, 2, 1))

        # set diagonal of distance matrix to 0
        proximity[:, torch.arange(proximity.shape[1]), torch.arange(proximity.shape[2])] = 0

        return proximity
    
    
    def getRMSD(self, reference_mol, positions):
        """
        Args :
            reference_mol : RDKit.Molecule
            positions : Tensor(n_atom, 3)
        """

        def optimizeWithFF(mol):

            mol = Chem.AddHs(mol, addCoords=True)
            AllChem.MMFFOptimizeMolecule(mol)
            mol = Chem.RemoveHs(mol)

            return mol

        n_atom = reference_mol.GetNumAtoms()

        test_cf = Chem.rdchem.Conformer(n_atom)
        for k in range(n_atom):
            test_cf.SetAtomPosition(k, positions[k].tolist())

        test_mol = copy.deepcopy(reference_mol)
        test_mol.RemoveConformer(0)
        test_mol.AddConformer(test_cf)

        if self.hparams.useFF:
            try:
                rmsd = AllChem.AlignMol(reference_mol, optimizeWithFF(test_mol))
            except:
                rmsd = AllChem.AlignMol(reference_mol, test_mol)
        else:
            rmsd = AllChem.AlignMol(reference_mol, test_mol)

        return rmsd
    
    
    def _draw_sample(self, mu, lsgms, masks):

        epsilon = torch.randn_like(lsgms)

        sample = torch.mul(torch.exp(0.5 * lsgms), epsilon)
        sample = torch.add(mu, sample)
        sample = torch.mul(sample, masks)

        return sample
    
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--alignment_type', type=str, default='kabsch', choices=['default', 'linear', 'kabsch'])
        parser.add_argument('--seed', type=int, default=1334, help='random seed for experiments')
        parser.add_argument('--use_X', action='store_true', help='use X as input for posterior of Z')
        parser.add_argument('--use_R', action='store_true', help='use R(X) as input for posterior of Z')
        parser.add_argument('--w_reg', type=float, default=1e-5, help='weight for conditional prior regularization')
        parser.add_argument('--refine_mom', type=float, default=0.99, help='momentum used for refinement')
        parser.add_argument('--refine_steps', type=int, default=0, help='number of refinement steps if requested')
        parser.add_argument('--useFF', action='store_true', help='use force field minimisation if testing')
        parser.add_argument('--dim_h', type=int, default=50, help='dimension of the hidden layer')
        parser.add_argument('--dim_f', type=int, default=100, 
                            help='dimension of the last hidden layer before embedding (mol or latent space)')
        parser.add_argument('--mpnn_steps', type=int, default=5, help='number of mpnn steps')
        return parser
    
class EmbedNode(nn.Module):
    
    def __init__(self, batch_size, n_max, node_dim, hidden_dim):
        
        super(EmbedNode, self).__init__()
        
        self.batch_size = batch_size
        self.n_max = n_max
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        
        self.FC_hidden = nn.Linear(node_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, nodes, masks):

        """
            Args :
                nodes : Tensor(batch_size, n_max, node_dim)
                masks : Tensor(batch_size, n_max, 1)
            Returns :
                nodes_embed : Tensor(batch_size, n_max, hidden_node_dim)
        """
        
        nodes_view = nodes.view(-1, self.node_dim)

        emb1 = torch.sigmoid(self.FC_hidden(nodes_view))
        emb2 = torch.tanh(self.FC_output(emb1))

        nodes_embed = emb2.view(-1, self.n_max, self.hidden_dim)
        nodes_embed = torch.mul(nodes_embed, masks)

        return nodes_embed
    
class EmbedEdge(nn.Module):
    
    def __init__(self, batch_size, n_max, edge_dim, hidden_dim):
        
        super(EmbedEdge, self).__init__()
        
        self.batch_size = batch_size
        self.n_max = n_max
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        
        self.FC_hidden = nn.Linear(edge_dim, 2 * hidden_dim)
        self.FC_output = nn.Linear(2 * hidden_dim, hidden_dim * hidden_dim)
        
    def forward(self, edges):

        """
            Args :
                edges :  Tensor(batch_size, n_max, n_max, edge_dim)
            Returns :
                edges_embed : Tensor(batch_size, n_max, n_max, hidden_dim, hidden_dim)
        """
        
        edges_view = edges.view(-1, self.edge_dim)

        emb1 = torch.sigmoid(self.FC_hidden(edges_view))
        emb2 = torch.tanh(self.FC_output(emb1))

        edges_embed = emb2.view(-1, self.n_max, self.n_max, self.hidden_dim, self.hidden_dim)

        return edges_embed
    
    
class MPNN(nn.Module):
    
    def __init__(self, batch_size, n_max, hidden_dim, mpnn_steps):
        
        super(MPNN, self).__init__()
        
        self.batch_size = batch_size
        self.n_max = n_max
        self.hidden_dim = hidden_dim
        self.mpnn_steps = mpnn_steps
        
        self.gru = torch.nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)
        
    def forward(self, edge_wgt, nodes_embed, mask): 

        """
            Args :
                edge_wgt : Tensor(batch_size, n_max, n_max, hidden_dim, hidden_dim)
                nodes_embed : Tensor(batch_size, n_max, hidden_node_dim)
                mask : Tensor(batch_size, n_max, 1)
            Returns :
                nodes_embed : Tensor(batch_size, n_max, hidden_node_dim)
        """
        
        for i in range(self.mpnn_steps):
        
            messages = self._compute_messages(edge_wgt, nodes_embed) # (batch_size, n_max, hidden_node_dim)
            nodes_embed = self._update_GRU(messages, nodes_embed, mask)

        return nodes_embed
    
    
    def _compute_messages(self, edge_wgt, nodes) :
        
        """
            Args :
                edge_wgt : Tensor(batch_size, n_max, n_max, hidden_dim, hidden_dim)
                nodes : Tensor(batch_size, n_max, hidden_node_dim)
            Returns :
                messages : Tensor(batch_size, n_max, hidden_node_dim)
        """
        
        weights = edge_wgt.view(-1, self.n_max * self.hidden_dim, self.hidden_dim)
        nodes = nodes.view(-1, self.hidden_dim, 1)

        messages = torch.matmul(weights, nodes)
        messages = messages.view(-1, self.n_max, self.n_max, self.hidden_dim)
        messages = messages.permute(0, 2, 3, 1)
        messages = messages.mean(3) / self.n_max

        return messages
    
    
    def _update_GRU(self, messages, nodes, mask):

        """
            Args :
                messages : Tensor(batch_size, n_max, hidden_dim)
                nodes : Tensor(batch_size, n_max, hidden_dim)
                mask : Tensor(batch_size, n_max, 1)
            Returns :
                nodes_next : Tensor(batch_size, n_max, hidden_dim)
        """

        #messages = messages.view(self.batch_size * self.n_max, 1, self.hidden_dim) was necessary for TF as it use a channel dim for multi RNN
        messages = messages.view(-1, self.hidden_dim) 
        nodes = nodes.view(-1, self.hidden_dim)
        
        nodes_next = self.gru(messages, nodes)

        nodes_next = nodes_next.view(-1, self.n_max, self.hidden_dim)
        nodes_next = torch.mul(nodes_next, mask)

        return nodes_next


class LatentNN(nn.Module):
    
    def __init__(self, batch_size, n_max, hidden_dim, dim_f, outdim):
        
        super(LatentNN, self).__init__()
        
        self.batch_size = batch_size
        self.n_max = n_max
        self.hidden_dim = hidden_dim
        self.dim_f = dim_f
        self.outdim = outdim
        
        self.dropout1 = torch.nn.Dropout(0.2)
        self.FC_hidden = nn.Linear(2*hidden_dim, dim_f)
        self.dropout2 = torch.nn.Dropout(0.2)
        #self.FC_hidden2 = nn.Linear(dim_f, dim_f)
        self.FC_output = nn.Linear(dim_f, outdim)
        
    def forward(self, nodes_embed, original_nodes_embed, mask):

        """
            Args :
                nodes_embed :  Tensor(batch_size, n_max, hidden_dim)
                original_nodes_embed :  Tensor(batch_size, n_max, hidden_dim)
            Returns :
                nodes_embed : Tensor(batch_size, n_max, outdim)
        """
        
        nodes_cat = torch.cat([nodes_embed, original_nodes_embed], 2)
        nodes_cat = nodes_cat.view(-1, nodes_cat.shape[2])
        
        nodes_cat = self.dropout1(nodes_cat)
        nodes_cat = torch.sigmoid(self.FC_hidden(nodes_cat))
        nodes_cat = self.dropout2(nodes_cat)
        # nodes_cat = torch.sigmoid(self.FC_hidden2(nodes_cat))
        nodes_cat = self.FC_output(nodes_cat)
        
        nodes_cat = nodes_cat.view(-1, self.n_max, self.outdim)
        nodes_cat = torch.mul(nodes_cat, mask)

        return nodes_cat
    
