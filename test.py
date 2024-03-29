import torch
import numpy as np
import pickle as pkl
import copy
import os

from rdkit import Chem
from rdkit.Chem import AllChem

def test(model, test_loader, val_num_samples=10, savepred_path='save_generated_mols/', savepermol=True, \
         refine_steps=0, useFF=False, device='cpu'):
    
    def pos_to_proximity(pos, mask):
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
    
    def getRMSD(reference_mol, positions, useFF=False):
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

        if useFF:
            try:
                rmsd = AllChem.AlignMol(reference_mol, optimizeWithFF(test_mol))
            except:
                rmsd = AllChem.AlignMol(reference_mol, test_mol)
        else:
            rmsd = AllChem.AlignMol(reference_mol, test_mol)

        return rmsd
    
    if not os.path.exists(savepred_path) :
        os.mkdir(savepred_path)
    
    # val_num_samples number of conformers to draw from prior 

    # val batch size is different from train batch size since sample each molecule {val_num_samples} times 
    val_batch_size = test_loader.batch_size # number of molecules per batch, to draw conformers from ; ie 2
    val_dataset_size = len(test_loader.dataset) # number of molecules in the dataset
    n_batch_val = int(val_dataset_size/val_batch_size) # number of batch ; ie 1500 if val_dataset_size = 3000
    assert (len(test_loader.dataset) % val_batch_size == 0)

    n_max = model.n_max
    
    rmsds_mol_mean = np.zeros(val_dataset_size)
    rmsds_mol_std = np.zeros(val_dataset_size)

    if savepred_path != None:
        if not savepermol:
            PX_preds = np.zeros(val_dataset_size, val_num_samples, n_max, 3)

    print ("testing model...")
    model.eval()

    with torch.no_grad():

        for batch_idx, batch in enumerate(test_loader):
            start_ = batch_idx * val_batch_size
            end_ = start_ + val_batch_size
            
            nodes, masks, edges, proximity, pos, mols = batch #D1 D2 D3 D4 D5
            nodes = nodes.to(device)
            masks = masks.to(device)
            edges = edges.to(device)
            proximity = proximity.to(device)
            pos = pos.to(device)
            masks = masks.unsqueeze(-1) # because dataloader squeezes the mask Tensor
            
            # repeat because we want val_batch_size (molecule) * val_num_samples (conformer per molecule)
            nodes = torch.repeat_interleave(nodes, val_num_samples, dim=0)
            masks = torch.repeat_interleave(masks, val_num_samples, dim=0)
            edges = torch.repeat_interleave(edges, val_num_samples, dim=0)
            proximity = torch.repeat_interleave(proximity, val_num_samples, dim=0)
                
            _, _, _, _, _, PX_pred = model(nodes, masks, edges, proximity, pos)

            X_pred = PX_pred
            for r in range(refine_steps):
                if use_X:
                    pos = X_pred
                if use_R:
                    proximity = pos_to_proximity(X_pred, mask)
                _, _, _, _, last_X_pred, _ = model(nodes, edges, mask, pos, proximity)
                X_pred = refine_mom * X_pred + (1-refine_mom) * last_X_pred

            rmsds=[]
            for j in range(X_pred.shape[0]):
                ms_v_index = int(j / val_num_samples)
                rmsd = getRMSD(mols[ms_v_index], X_pred[j], useFF)
                rmsds.append(rmsd)

            rmsds = np.array(rmsds)
            rmsds = np.reshape(rmsds, (val_batch_size, val_num_samples))
            rmsds_mol_mean[start_:end_] = np.mean(rmsds, axis=1)
            rmsds_mol_std[start_:end_] = np.std(rmsds, axis=1)

            if savepred_path != None:
                if not savepermol:
                    PX_preds[start_:end_] = PX_pred.view(val_batch_size, val_num_samples, n_max, 3)
            
            # save results per molecule if request
            if savepermol:
                curr_X_preds = copy.deepcopy(X_pred).view(val_batch_size, val_num_samples, n_max, 3)
                for mol_i in range(0, val_batch_size):
                    save_dict_mol = {'rmsd': rmsds[mol_i], 'pred': curr_X_preds[mol_i]}
                    pkl.dump(save_dict_mol, \
                        open(os.path.join(savepred_path, 'mol_{}_refined.p'.format(mol_i+start_)), 'wb'))

        print ("val scores: mean is {} , std is {}".format(np.mean(rmsds_mol_mean), np.mean(rmsds_mol_std)))
        
        if savepred_path != None:
            if not savepermol:
                print ("saving neural net predictions into {}".format(savepred_path))
                pkl.dump(PX_preds, open(savepred_path + 'PX_preds', 'wb'))

        return np.mean(rmsds_mol_mean), np.mean(rmsds_mol_std)
    
    
    def getRMS(self, prb_mol, ref_pos, useFF=False):

        def optimizeWithFF(mol):

            molf = Chem.AddHs(mol, addCoords=True)
            AllChem.MMFFOptimizeMolecule(molf)
            molf = Chem.RemoveHs(molf)

            return molf

        n_est = prb_mol.GetNumAtoms()

        ref_cf = Chem.rdchem.Conformer(n_est)
        for k in range(n_est):
            ref_cf.SetAtomPosition(k, ref_pos[k].tolist())

        ref_mol = copy.deepcopy(prb_mol)
        ref_mol.RemoveConformer(0)
        ref_mol.AddConformer(ref_cf)

        if useFF:
            try:
                res = AllChem.AlignMol(prb_mol, optimizeWithFF(ref_mol))
            except:
                res = AllChem.AlignMol(prb_mol, ref_mol)
        else:
            res = AllChem.AlignMol(prb_mol, ref_mol)

        return res