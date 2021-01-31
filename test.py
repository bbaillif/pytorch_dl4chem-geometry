import torch
import numpy as np
import pickle as pkl
import copy

from rdkit import Chem
from rdkit.Chem import AllChem

def test(model, test_loader, molsup_val, val_num_samples=10, debug=False, savepred_path=None, savepermol=False, \
         refine_steps=0, useFF=False):

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
    
    # val_num_samples number of conformers to draw from prior 

    # val batch size is different from train batch size since we use multiple samples
    val_batch_size = test_loader.batch_size # number of molecules to draw conformers from ; ie 2
    n_batch_val = int(len(test_loader.dataset)/val_batch_size) # 1500 if D1_v = 3000
    assert (len(test_loader.dataset) % val_batch_size == 0)

    val_size = len(test_loader.dataset)
    valscores_mean = np.zeros(val_size)
    valscores_std = np.zeros(val_size)

    if savepred_path != None:
        if not savepermol:
            pred_v = np.zeros(D1_v.shape[0], val_num_samples, n_max, 3)

    print ("testing model...")
    model.eval()

    with torch.no_grad():

        for batch_idx, batch in enumerate(test_loader):
            start_ = batch_idx * val_batch_size
            end_ = start_ + val_batch_size

            print('Test batch number' + str(batch_idx))
            
            nodes, masks, edges, proximity, pos = batch #D1 D2 D3 D4 D5
            masks = masks.unsqueeze(-1) # because dataloader squeezes the mask Tensor
            
            # repeat because we want val_batch_size (molecule) * val_num_samples (conformer per molecule)
            nodes = torch.repeat_interleave(nodes, val_num_samples, dim=0)
            masks = torch.repeat_interleave(masks, val_num_samples, dim=0)
            edges = torch.repeat_interleave(edges, val_num_samples, dim=0)
            proximity = torch.repeat_interleave(proximity, val_num_samples, dim=0)

            if debug:
                print (i, len(test_loader))
                
            _, _, _, _, _, PX_pred = model(nodes, masks, edges, proximity, pos)

            if savepred_path != None:
                if not savepermol:
                    pred_v[start_:end_] = PX_pred.view(val_batch_size, val_num_samples, n_max, 3)

            X_pred = PX_pred
            for r in range(refine_steps):
                if use_X:
                    pos = X_pred
                if use_R:
                    proximity = pos_to_proximity(X_pred, mask)
                _, _, _, _, last_X_pred, _ = model(nodes, edges, mask, pos, proximity)
                X_pred = refine_mom * X_pred + (1-refine_mom) * last_X_pred

            valrmsd=[]
            for j in range(X_pred.shape[0]):
                ms_v_index = int(j / val_num_samples) + start_
                rmsd = getRMSD(molsup_val[ms_v_index], X_pred[j], useFF)
                valrmsd.append(rmsd)

            valrmsd = np.array(valrmsd)
            valrmsd = np.reshape(valrmsd, (val_batch_size, val_num_samples))
            valrmsd_mean = np.mean(valrmsd, axis=1)
            valrmsd_std = np.std(valrmsd, axis=1)

            valscores_mean[start_:end_] = valrmsd_mean
            valscores_std[start_:end_] = valrmsd_std

            # save results per molecule if request
            if savepermol:
                pred_curr = copy.deepcopy(X_pred).view(val_batch_size, val_num_samples, n_max, 3)
                for tt in range(0, val_batch_size):
                    save_dict_tt = {'rmsd': valrmsd[tt], 'pred': pred_curr[tt]}
                    pkl.dump(save_dict_tt, \
                        open(os.path.join(savepred_path, 'mol_{}_neuralnet.p'.format(tt+start_)), 'wb'))

        print ("val scores: mean is {} , std is {}".format(np.mean(valscores_mean), np.mean(valscores_std)))
        if savepred_path != None:
            if not savepermol:
                print ("saving neural net predictions into {}".format(savepred_path))
                pkl.dump(pred_v, open(savepred_path, 'wb'))

        return np.mean(valscores_mean), np.mean(valscores_std)