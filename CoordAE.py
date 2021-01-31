import torch

from torch import nn

# TODO : try to fix masking (true mask and virtual node)
# Naming of variables (ie original nodes embed vs nodes embed), keep the same terminology

# 3D Coordinates autoencoder model
class CoordAE(nn.Module):

    def __init__(self, n_max, dim_node, dim_edge, dim_h, dim_f, \
                batch_size, mpnn_steps=5, alignment_type='default', tol=1e-5, \
                use_X=True, use_R=True, seed=0, \
                refine_steps=0, refine_mom=0.99):
        
        super(CoordAE, self).__init__()
        
        self.mpnn_steps = mpnn_steps
        self.n_max = n_max
        self.dim_node = dim_node
        self.dim_edge = dim_edge
        self.dim_h = dim_h
        self.dim_f = dim_f
        self.batch_size = batch_size
        self.tol = tol
        self.refine_steps = refine_steps
        self.refine_mom = refine_mom
        self.use_X = use_X
        self.use_R = use_R
            
        #self.G not included
        #placeholders not included
        #prior_T not included
        
        # find a way to define self.mask
        
        # ADD SELF IN ARGS FOR FOLLOWING LINES
        self.embed_nodes = EmbedNode(batch_size, n_max, dim_node, dim_h)
        
        # Prior Z
        self.edge_nn_prior_z = EdgeNN(batch_size, n_max, dim_edge + 1, dim_h)
        self.mpnn_prior_z = MPNN(batch_size, n_max, dim_h, mpnn_steps)
        self.latent_nn_prior_z = LatentNN(batch_size, n_max, dim_h, dim_f, 2*dim_h)
        
        # Post Z
        if self.use_R :
            self.edge_nn_post_z = EdgeNN(batch_size, n_max, dim_edge + 2, dim_h)
        else :
            self.edge_nn_post_z = EdgeNN(batch_size, n_max, dim_edge + 1, dim_h)
            
        if use_X:
            self.embed_nodes_pos = EmbedNode(batch_size, n_max, dim_node + 3, dim_h)
            
        self.mpnn_post_z = MPNN(batch_size, n_max, dim_h, mpnn_steps)
        self.latent_nn_post_z = LatentNN(batch_size, n_max, dim_h, dim_f, 2*dim_h)
        
        # Post X
        self.edge_nn_post_x = EdgeNN(batch_size, n_max, dim_edge + 1, dim_h)
        self.mpnn_post_x = MPNN(batch_size, n_max, dim_h, mpnn_steps)
        self.latent_nn_post_x = LatentNN(batch_size, n_max, dim_h, dim_f, 3)
        
        # Post X det
        self.edge_nn_post_x_det = EdgeNN(batch_size, n_max, dim_edge + 1, dim_h)
        self.mpnn_post_x_det = MPNN(batch_size, n_max, dim_h, mpnn_steps)
        self.latent_nn_post_x_det = LatentNN(batch_size, n_max, dim_h, dim_f, 3)
        
        # Pred X
        self.edge_nn_pred_x = EdgeNN(batch_size, n_max, dim_edge + 1, dim_h)
        self.mpnn_pred_x = MPNN(batch_size, n_max, dim_h, mpnn_steps)
        self.latent_nn_pred_x = LatentNN(batch_size, n_max, dim_h, dim_f, 3)
        
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

        n_atom = masks.permute(0, 2, 1).sum(2) # (batch_size, 1)

        tiled_n_atom = n_atom.view(self.batch_size, 1, 1, 1).repeat(1, self.n_max, self.n_max, 1) # (batch_size, n_max, n_max, 1)

        # Isn't there a better way to add n_atom in edge features ?
        edge_2 = torch.cat([edges, tiled_n_atom], 3) # (batch_size, n_max, nmax, dim_edge + 1)


        # p(Z|G) -- prior of Z

        priorZ_edge_wgt = self.edge_nn_prior_z(edge_2) #[batch_size, n_max, n_max, dim_h, dim_h]
        priorZ_hidden = self.mpnn_prior_z(priorZ_edge_wgt, nodes_embed, masks) # (batch_size, n_max, dim_h), nodes_embed like
        priorZ_out = self.latent_nn_prior_z(priorZ_hidden, nodes_embed, masks) # (batch_size, n_max, 2*dim_h)

        priorZ_mu, priorZ_lsgms = priorZ_out.split([self.dim_h, self.dim_h], 2)
        priorZ_sample = self._draw_sample(priorZ_mu, priorZ_lsgms, masks)


        # q(Z|R(X),G) -- posterior of Z, used R instead of X as input for simplicity, should be updated

        if self.use_R:
            proximity_view = proximity.view(self.batch_size, self.n_max, self.n_max, 1)
            edge_cat = torch.cat([edge_2, proximity_view], 3) #[batch_size, n_max, n_max, dim_edge + 2]
            postZ_edge_wgt = self.edge_nn_post_z(edge_cat) #[batch_size, n_max, n_max, dim_h, dim_h]
        else:
            postZ_edge_wgt = self.edge_nn_post_z(edge_2) 

        if self.use_X:
            nodes_pos = torch.cat([nodes, pos], 2) # (batch_size, n_max, dim_node + 3)
            nodes_pos_embed = self.embed_nodes_pos(nodes_pos, masks)
            postZ_hidden = self.mpnn_post_z(postZ_edge_wgt, nodes_pos_embed, masks)
        else:
            postZ_hidden = self.mpnn_post_z(postZ_edge_wgt, nodes_embed, masks)

        postZ_out = self.latent_nn_prior_z(postZ_hidden, nodes_embed, masks)

        postZ_mu, postZ_lsgms = postZ_out.split([self.dim_h, self.dim_h], 2)
        postZ_sample = self._draw_sample(postZ_mu, postZ_lsgms, masks)


        # p(X|Z,G) -- posterior of X

        X_edge_wgt = self.edge_nn_post_x(edge_2) #[batch_size, n_max, n_max, dim_h, dim_h]
        X_hidden = self.mpnn_post_x(X_edge_wgt, postZ_sample + nodes_embed, masks)
        X_pred = self.latent_nn_post_x(X_hidden, nodes_embed, masks)


        # p(X|Z,G) -- posterior of X without sampling from latent space
        # used for iterative refinement of predictions ; det stands for deterministic

        X_edge_wgt_det = self.edge_nn_post_x_det(edge_2) #[batch_size, n_max, n_max, dim_h, dim_h]
        X_hidden_det = self.mpnn_post_x_det(X_edge_wgt_det, postZ_mu + nodes_embed, masks)
        X_pred_det = self.latent_nn_post_x_det(X_hidden_det, nodes_embed, masks)


        # Prediction of X with p(Z|G) in the test phase

        PX_edge_wgt = self.edge_nn_pred_x(edge_2) #[batch_size, n_max, n_max, dim_h, dim_h]
        PX_hidden = self.mpnn_pred_x(PX_edge_wgt, priorZ_sample + nodes_embed, masks)
        PX_pred = self.latent_nn_pred_x(PX_hidden, nodes_embed, masks)

        return postZ_mu, postZ_lsgms, priorZ_mu, priorZ_lsgms, X_pred, PX_pred

    def _draw_sample(self, mu, lsgms, masks):

        epsilon = torch.randn_like(lsgms)

        sample = torch.mul(torch.exp(0.5 * lsgms), epsilon)
        sample = torch.add(mu, sample)
        sample = torch.mul(sample, masks)

        return sample
    
    
# There may be a way to merge EmbedNode and EdgeNN
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
        
        nodes_view = nodes.view(self.batch_size * self.n_max, nodes.shape[2])

        emb1 = torch.sigmoid(self.FC_hidden(nodes_view))
        emb2 = torch.tanh(self.FC_output(emb1))

        nodes_embed = emb2.view(self.batch_size, self.n_max, self.hidden_dim)
        nodes_embed = torch.mul(nodes_embed, masks)

        return nodes_embed
    
class EdgeNN(nn.Module):
    
    def __init__(self, batch_size, n_max, edge_dim, hidden_dim):
        
        super(EdgeNN, self).__init__()
        
        self.batch_size = batch_size
        self.n_max = n_max
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        
        self.FC_hidden = nn.Linear(edge_dim, 2 * hidden_dim)
        self.FC_output = nn.Linear(2 * hidden_dim, hidden_dim * hidden_dim)
        
    def forward(self, edges):

        """
            Args :
                edges :  Tensor(batch_size, n_max, node_dim)
            Returns :
                edges_embed : Tensor(batch_size, n_max, n_max, hidden_dim, hidden_dim)
        """
        
        edges_view = edges.view(self.batch_size * self.n_max * self.n_max, edges.shape[3])

        emb1 = torch.sigmoid(self.FC_hidden(edges_view))
        emb2 = torch.tanh(self.FC_output(emb1))

        edges_embed = emb2.view(self.batch_size, self.n_max, self.n_max, self.hidden_dim, self.hidden_dim)

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
    
    def _update_GRU(self, messages, nodes, mask):

        """
            Args :
                messages : Tensor(batch_size, n_max, hidden_dim)
                nodes : Tensor(batch_size, n_max, hidden_dim)
                mask : Tensor(batch_size, n_max, 1)
            Returns :
                nodes_next : Tensor(batch_size, n_max, hidden_dim)
        """

        #messages = messages.view(self.batch_size * self.n_max, 1, self.hidden_dim) 
        messages = messages.view(self.batch_size * self.n_max, self.hidden_dim) 
        nodes = nodes.view(self.batch_size * self.n_max, self.hidden_dim)
        
        nodes_next = self.gru(messages, nodes)

        nodes_next = nodes_next.view(self.batch_size, self.n_max, self.hidden_dim)
        nodes_next = torch.mul(nodes_next, mask)

        return nodes_next
    
    def _compute_messages(self, edge_wgt, nodes) :
        
        """
            Args :
                edge_wgt : Tensor(batch_size, n_max, hidden_dim)
                nodes : Tensor(batch_size, n_max, hidden_node_dim)
            Returns :
                messages : Tensor(batch_size, n_max, hidden_node_dim)
        """
        
        weights = edge_wgt.view(self.batch_size * self.n_max, self.n_max * self.hidden_dim, self.hidden_dim)
        nodes = nodes.view(self.batch_size * self.n_max, self.hidden_dim, 1)

        messages = torch.matmul(weights, nodes)
        messages = messages.view(self.batch_size, self.n_max, self.n_max, self.hidden_dim)
        messages = messages.permute(0, 2, 3, 1)
        messages = messages.mean(3) / self.n_max

        return messages
    

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
        nodes_cat = nodes_cat.view(self.batch_size * self.n_max, nodes_cat.shape[2])
        
        nodes_cat = self.dropout1(nodes_cat)
        nodes_cat = torch.sigmoid(self.FC_hidden(nodes_cat))
        nodes_cat = self.dropout2(nodes_cat)
        # nodes_cat = torch.sigmoid(self.FC_hidden2(nodes_cat))
        nodes_cat = self.FC_output(nodes_cat)
        
        nodes_cat = nodes_cat.view(self.batch_size, self.n_max, self.outdim)
        nodes_cat = torch.mul(nodes_cat, mask)

        return nodes_cat
    
