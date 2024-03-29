import torch
import numpy as np

# Add underscore to internal functions

class MSDScorer(object) :
    
    def __init__(self, alignment_type='kabsch2', tol=1e-5):
        self.alignment_type = alignment_type
        self.tol = tol
        
        if alignment_type == 'linear':
            self.msd_func = self.linear_transform_msd
        elif alignment_type == 'kabsch':
            self.msd_func = self.kabsch_msd
        elif alignment_type == 'default':
            self.msd_func = self.mol_msd
        elif alignment_type == 'kabsch2':
            self.msd_func = self.kabsch_rmsd2
        elif alignment_type == 'kabsch3':
            self.msd_func = self.kabsch_rmsd3
        
        
    def score(self, X_pred, pos, masks=None) :
        return self.msd_func(X_pred, pos, masks)
        
        
    # Does not work, cause the model to collapse to nan weights
    def kabsch_rmsd2(self, frames, targets, masks):
        rmsds = []
        for i in range(frames.shape[0]) :
            frame = frames[i]
            target = targets[i]
            
            mask = masks[i]
            frame = self.do_mask(frame, mask)
            target = self.do_mask(target, mask)
            
            frame_T = frame.T
            target_T = target.T
            frame_cent = frame_T - frame_T.mean(1).view(3, 1)
            target_cent = target_T - target_T.mean(1).view(3, 1).detach()
            
            cov = torch.matmul(frame_cent, target_cent.T)
            try:
                U, S, V = torch.svd(cov, some=False)
            except RuntimeError:
                print("SVD failed to converge")
                rmsds.append(torch.tensor([1]).type_as(frame).detach())
            else :
                d = torch.tensor([
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, torch.det(torch.matmul(V, U.transpose(0, 1)))]
                ]).type_as(frame).detach()
                rot = torch.matmul(torch.matmul(V, d), U.transpose(0, 1))
                frame_cent_rot = torch.matmul(rot, frame_cent)

                diffs = frame_cent_rot - target_cent
                msd = (diffs ** 2).sum() / diffs.size(1)
                rmsds.append(msd.sqrt())
        
        loss = torch.stack(rmsds)
        return loss
        
        
    def kabsch_msd(self, frames, targets, masks):
        losses = []
        for i in range(len(frames)):
            frame = frames[i]
            target = targets[i]
            if masks != None :
                mask = masks[i]
            else :
                mask = None
            target_cent = target - self.torch_centroid_masked(target, mask)
            frame_cent = frame - self.torch_centroid_masked(frame, mask)
            losses.append(self.torch_kabsch_rmsd_masked(target_cent.detach(), frame_cent, mask))

        loss = torch.stack(losses)
        return loss

    
    def optimal_rotational_quaternion(self, r):
        """Just need the largest eigenvalue of this to minimize RMSD over rotations

        References
        ----------
        [1] http://dx.doi.org/10.1002/jcc.20110
        """
        return [
            [r[0][0] + r[1][1] + r[2][2], r[1][2] - r[2][1], r[2][0] - r[0][2], r[0][1] - r[1][0]],
            [r[1][2] - r[2][1], r[0][0] - r[1][1] - r[2][2], r[0][1] + r[1][0], r[0][2] + r[2][0]],
            [r[2][0] - r[0][2], r[0][1] + r[1][0], -r[0][0] + r[1][1] - r[2][2], r[1][2] + r[2][1]],
            [r[0][1] - r[1][0], r[0][2] + r[2][0], r[1][2] + r[2][1], -r[0][0] - r[1][1] + r[2][2]],
        ]
    
    
    def squared_deviation(self, frame, target):
        """Calculate squared deviation (n_atoms * RMSD^2) from `frame` to `target`
        First we compute `R` which is the ordinary cross-correlation of xyz coordinates.
        Turns out you can do a bunch of quaternion math to find an eigen-expression for finding optimal
        rotations. There aren't quaternions in tensorflow, so we use the handy formula for turning
        quaternions back into 4-matrices. This is the `F` matrix. We find its leading eigenvalue
        to get the MSD after optimal rotation. Note: *finding* the optimal rotation requires the values
        and vectors, but we don't care.

        Parameters
        ----------
        frame, target : Tensor, shape=(n_atoms, 3)
            Calculate the MSD between these two frames

        Returns
        -------
        sd : Tensor, shape=(0,)
            Divide by number of atoms and take the square root for RMSD
        """
        R = torch.matmul(frame.T, target)
        R_parts = [torch.unbind(t) for t in torch.unbind(R)]
        F_parts = self.optimal_rotational_quaternion(R_parts)
        F = torch.Tensor(F_parts)
        vals, vecs = torch.symeig(F, eigenvectors=True)
        # This isn't differentiable for some godforsaken reason.
        # vals = tf.self_adjoint_eigvals(F, name='vals')
        lmax = torch.unbind(vals)[-1]
        sd = torch.sum(frame ** 2 + target ** 2) - 2 * lmax
        return sd
    
    
    # https://towardsdatascience.com/tensorflow-rmsd-using-tensorflow-for-things-it-was-not-designed-to-do-ada4c9aa0ea2
    # https://github.com/mdtraj/tftraj
    def mol_msd(self, frames, targets, masks):
        frames -= frames.mean(1, keepdim=True)
        targets -= targets.mean(1, keepdim=True)

        loss = torch.stack([self.squared_deviation( self.do_mask(frames[i], masks[i]), self.do_mask(targets[i], masks[i]) ) for i in range(len(frames))], 0)
        return loss / masks.sum((1,2))

    
    # broken if mask is not a full ones tensor
    def linear_transform_msd(self, frames, targets, masks):
        def linearly_transform_frames(padded_frames, padded_targets):
            u, s, v = torch.svd(padded_frames)
            tol = 1e-7
            atol = s.max() * tol
            s = torch.masked_select(s, s > atol) # this step seems weird because it reduces the shape of s, that will later be matmul to u matrix
            s_inv = torch.diag(1. / s)
            pseudo_inverse = torch.matmul(v, torch.matmul(s_inv, u.T))

            weight_matrix = torch.matmul(padded_targets, pseudo_inverse)
            transformed_frames = torch.matmul(weight_matrix, padded_frames)
            return transformed_frames
        
        padded_frames = torch.nn.functional.pad(frames, (0, 1), 'constant', 1)
        padded_targets = torch.nn.functional.pad(targets, (0, 1), 'constant', 1)

        mask_matrices = []
        for i in range(len(frames)):
            mask_matrix = torch.diag(masks[i].view(-1))
            mask_matrices.append(mask_matrix)
        #mask_matrix = tf.diag(tf.reshape(masks, [self.batch_size, -1]))
        mask_tensor = torch.stack(mask_matrices)
        masked_frames = torch.matmul(mask_tensor, padded_frames)
        masked_targets = torch.matmul(mask_tensor, padded_targets)
        transformed_frames = []
        for i in range(len(frames)):
            transformed_frames.append(linearly_transform_frames(masked_frames[i], masked_targets[i]))
        transformed_frames = torch.stack(transformed_frames)
        #transformed_frames = linearly_transform_frames(masked_frames, masked_targets)
        mse_loss = torch.nn.MSELoss()
        loss = mse_loss(transformed_frames, masked_targets)

        return loss
        
        
    def torch_kabsch(self, P, Q):
        # calculate covariance matrix
        C = torch.matmul(P.T, Q)
        
        V, S, W = torch.svd(C, some=False)
        
        def adjoint(W) :
            W = torch.transpose(W, -2, -1)
            W = torch.conj(W)
            return W
        
        W = adjoint(W)
            
        m1 = torch.ones((3,), dtype=torch.float32).type_as(P).detach()
        m1[-1] = -m1[-1]

        m2 = torch.ones((3,3), dtype=torch.float32).type_as(P).detach()
        m2[:,-1] = -m2[:,-1]

        d = torch.det(V) * torch.det(W)
        S = torch.where(d < 0., S * m1, S)
        V = torch.where(d < 0., V * m2, V)
        # Rotation matrix U
        U = torch.matmul(V, W)
        return U

    # maybe I could implement a batch RMSE implementation
    # N could be handled by taking the number of rows different from [0 0 0] 
    
    
    def torch_rmsd_masked(self, V, W, N=None):
        """
        Compute the RMSD between the two coordinates matrices V and W.
        N (int) is the number of atoms having coordinates (selected via masking in the workflow)
        Args :
            V : Tensor(n_max, 3)
            W : Tensor(n_max, 3)
            N : int
        Returns :
            RMSD : float
        """
        if N is None :
            N = torch.Tensor([V.shape[0]]).type_as(V)
        SE = (V - W) ** 2 # SE = Squared Error
        MSE = SE.sum() / N.float() # MSE = Mean Squared Error
        return torch.sqrt(MSE) # RMSE = Mean Squared Error

    
    def torch_kabsch_rotate(self, P, Q):
        U = self.torch_kabsch(P, Q) # rotate matrix P
        return torch.matmul(P, U)

    
    def torch_kabsch_rmsd_masked(self, P, Q, mask=None):
        N = None
        if mask != None :
            N = mask.sum()
            mask_mat = torch.diag(mask.view((-1,)))
            P = torch.matmul(mask_mat, P) + self.tol
            Q = torch.matmul(mask_mat, Q) + self.tol
        P_transformed = self.torch_kabsch_rotate(P, Q)
        return self.torch_rmsd_masked(P_transformed, Q, N)

    
    def torch_centroid_masked(self, P, mask=None):
        N = torch.Tensor([P.shape[0]]).type_as(P)
        if mask != None : # mask P
            N = mask.sum()
            mask_mat = torch.diag(mask.view((-1,)))
            
            P = torch.matmul(mask_mat, P) + self.tol # necesary steps to apply mask to prediction
        return P.sum(0, keepdim=True) / N.float()

    
    def do_mask(self, vec, mask):
        return vec[torch.gt(mask, 0.5).view((mask.shape[0],))]
    
    
    # To do properly
    def tests(self) :
        A = torch.tensor([[1., 1., 1], [2., 2., 2], [1.5, 3., 4.5]], dtype=torch.float)
        R0 = torch.tensor([[1, 0, 0], [0, np.cos(60), -np.sin(60)], [0, np.sin(60), np.cos(60)]], dtype=torch.float)
        B = (R0.mm(A.T)).T
        t0 = torch.tensor([3., 3., 3.])
        B += t0
        A_batch = A.unsqueeze(0)
        B_batch = B.unsqueeze(0)

        mask = torch.ones(A_batch.shape[0], A_batch.shape[1], 1)
        print(self.kabsch_msd(A_batch, B_batch, mask))
        print(self.kabsch_rmsd2(A_batch, B_batch, mask))
        print(self.kabsch_rmsd3(A_batch, B_batch, mask))
        print(self.mol_msd(A_batch, B_batch, mask))
        print(self.linear_transform_msd(A_batch, B_batch, mask))

        mask[:,-1] = 0
        print(self.kabsch_msd(A_batch, B_batch, mask))
        print(self.mol_msd(A_batch, B_batch, mask))
        
        A = torch.tensor([[1., 1., 1], [2., 2., 2], [1.5, 3., 4.5]], dtype=torch.float)
        B = torch.tensor([[1.5, 3., 4.5], [2., 2., 2], [1., 1., 1]], dtype=torch.float)
        A_batch = A.unsqueeze(0)
        B_batch = B.unsqueeze(0)

        mask = torch.ones(A_batch.shape[0], A_batch.shape[1], 1)
        print(f"1 {self.kabsch_msd(A_batch, B_batch, mask)}")
        print(f"2 {self.kabsch_rmsd2(A_batch, B_batch, mask)}")
        print(f"3 {self.kabsch_rmsd3(A_batch, B_batch, mask)}")
        print(self.mol_msd(A_batch, B_batch, mask))
        print(self.linear_transform_msd(A_batch, B_batch, mask))
        # linear is out because the points are not supposed to be the same, RMSD should be positive
        # the order of the points count

        mask[:,-1] = 0
        print(self.kabsch_msd(A_batch, B_batch, mask))
        print(self.mol_msd(A_batch, B_batch, mask))