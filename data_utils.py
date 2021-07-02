import torch
import pytorch_lightning as pl
import pickle as pkl
import numpy as np
import h5py
from torch.utils.data import random_split, DataLoader, Dataset
from torch.utils.data.dataloader import default_collate 

class PickleDataset(torch.utils.data.Dataset):
    """ Reads and collects data from an pickle files with five "datasets"
    """
    def __init__(self, nodes, masks, edges, proximities, positions, mols):

        self.nodes = nodes
        self.masks = masks
        self.edges = edges
        self.proximities = proximities
        self.positions = positions
        self.mols = mols

        # get the number of elements in the dataset
        self.n_graphs = self.nodes.shape[0]

    def __getitem__(self, idx):
        # returns specific graph elements
        nodes_i = torch.from_numpy(self.nodes[idx]).type(torch.float32)
        masks_i = torch.from_numpy(self.masks[idx]).type(torch.float32)
        edges_i = torch.from_numpy(self.edges[idx]).type(torch.float32)
        proximities_i = torch.from_numpy(self.proximities[idx]).type(torch.float32)
        positions_i = torch.from_numpy(self.positions[idx]).type(torch.float32)
        mols_i = self.mols[idx]

        return nodes_i, masks_i, edges_i, proximities_i, positions_i, mols_i

    def __len__(self):
        # returns the number of graphs in the dataset
        return self.n_graphs
    

class HDFDataset(torch.utils.data.Dataset):
    """ Reads and collects data from an HDF file with three HDF "datasets":
    "nodes", "edges", and "APDs".
    """
    def __init__(self, filename, mols):

        self.filename = filename

        hdf_file = h5py.File(self.filename, "r+", swmr=True)

        # load each HDF dataset
        self.nodes = hdf_file.get("nodes")
        self.masks = hdf_file.get("masks")
        self.edges = hdf_file.get("edges")
        self.proximities = hdf_file.get("dist_mats")
        self.positions = hdf_file.get("pos")
        self.mols = mols

        # get the number of elements in the dataset
        self.n_subgraphs = self.nodes.shape[0]

    def __getitem__(self, idx):
        # returns specific graph elements
        nodes_i = torch.from_numpy(self.nodes[idx]).type(torch.float32)
        masks_i = torch.from_numpy(self.masks[idx]).type(torch.float32)
        edges_i = torch.from_numpy(self.edges[idx]).type(torch.float32)
        proximities_i = torch.from_numpy(self.proximities[idx]).type(torch.float32)
        positions_i = torch.from_numpy(self.positions[idx]).type(torch.float32)
        mols_i = self.mols[idx]

        return nodes_i, masks_i, edges_i, proximities_i, positions_i, mols_i

    def __len__(self):
        # returns the number of graphs in the dataset
        return self.n_subgraphs
    
    
class MoleculeDataModule(pl.LightningDataModule):

    def __init__(self, 
                 dataset: str = 'QM9',
                 data_dir: str = '/home/bb596/rds/hpc-work/dl4chem/',
                 batch_size: int = 20,
                 val_num_sample: int = 10,
                 val_ratio: float = 0.05,
                 test_ratio: float = 0.05,
                 filename: str ="csd_dataset.hdf5"):
        
        super().__init__()
        
        self.data_dir = data_dir
        self.dataset = dataset
        
        if dataset == 'QM9' :
            self.n_max = 9
        else : # COD or CSD
            self.n_max = 50
            
            
        self.batch_size = batch_size
        self.val_num_sample = val_num_sample
        self.val_batch_size = batch_size // val_num_sample
            
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.train_ratio = 1 - self.val_ratio - self.test_ratio
            
        self.molset_fname = self.data_dir + self.dataset + '_molset_' + str(self.n_max) + '.p'
        if dataset == 'CSD' :
            self.filename = filename
        else :
            self.nodes_fname = self.data_dir + self.dataset + '_nodes_' + str(self.n_max) + '.p'
            self.masks_fname = self.data_dir + self.dataset + '_masks_' + str(self.n_max)+'.p'
            self.edges_fname = self.data_dir + self.dataset + '_edges_' + str(self.n_max)+'.p'
            self.dist_mats_fname = self.data_dir + self.dataset + '_dist_mats_' + str(self.n_max)+'.p'
            self.positions_fname = self.data_dir + self.dataset + '_positions_' + str(self.n_max)+'.p'
        
#         if dataset == 'CSD' : requires a lot of RAM
#             self.edges_fnames = [self.data_dir + self.dataset + '_edges_' + str(self.n_max)+'_'+str(i)+'.p' for i in range(11)]
        
    def setup(self, stage=None):
        
        [mols, smiles] = pkl.load(open(self.molset_fname,'rb'))
        mols = np.array(mols)
        
        if self.dataset == 'CSD' :
            # TODO : split train val test in another script, create directly the 3 datasets
            self.molecule_dataset = HDFDataset(self.data_dir + self.filename, mols)
        else : # pickled files
            nodes = pkl.load(open(self.nodes_fname,'rb')).todense()
            masks = pkl.load(open(self.masks_fname,'rb')).todense()

#             if self.dataset == 'CSD' : requires a lot of RAM
#                 dist_mats = dist_mats.todense()
#                 edges = [pkl.load(open(fname,'rb')).todense() for fname in self.edges_fnames]
#                 edges = np.vstack(edges)
#             else :
            edges = pkl.load(open(self.edges_fname,'rb')).todense()
            dist_mats = pkl.load(open(self.dist_mats_fname,'rb'))
            pos = pkl.load(open(self.positions_fname,'rb'))

            self.molecule_dataset = PickleDataset(nodes, masks, edges, dist_mats, pos, mols)
        
            dataset_size = len(self.molecule_dataset)
            self.val_size = int(dataset_size * self.val_ratio)
            self.test_size = int(dataset_size * self.test_ratio)
            self.train_size = dataset_size - self.val_size - self.test_size

            lengths = [self.train_size, self.val_size, self.test_size]
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.molecule_dataset, lengths, generator=torch.Generator().manual_seed(42))
        
    def train_dataloader(self):
        if self.dataset == 'CSD' :
            return BlockDataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self._custom_collate, num_workers=12, pin_memory=True) 
        else :
            return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self._custom_collate, num_workers=12, pin_memory=True) 
    
    
    def val_dataloader(self):
        if self.dataset == 'CSD' :
            return BlockDataLoader(self.val_dataset, batch_size=self.val_batch_size, collate_fn=self._custom_collate, num_workers=12, pin_memory=True)
        else :
            return DataLoader(self.val_dataset, batch_size=self.val_batch_size, collate_fn=self._custom_collate, num_workers=12, pin_memory=True)

    
    def test_dataloader(self):
        if self.dataset == 'CSD' :
            return BlockDataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self._custom_collate, num_workers=12, pin_memory=True)
        else :
            return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self._custom_collate, num_workers=12, pin_memory=True)
    
    
    def _custom_collate(self, batch):
        molecule_tensors = [b[:-1] for b in batch]
        mols = [b[-1] for b in batch]
        return default_collate(molecule_tensors), mols
    
    
### Only relevent if using h5py datasets, if large dataset cannot be loaded into RAM
class BlockDataLoader(torch.utils.data.DataLoader):
    """ Main `DataLoader` class which has been modified so as to read training
    data from disk in blocks, as opposed to a single line at a time (as is done
    in the original `DataLoader` class).
    """
    def __init__(self, dataset, batch_size=20, block_size=10000,
                shuffle=False, num_workers=0, pin_memory=True, collate_fn=default_collate):

        # define variables to be used throughout dataloading
        self.dataset = dataset        # `PickleDataset` object
        self.batch_size = batch_size  # `int`
        self.block_size = block_size  # `int`
        self.shuffle = shuffle        # `bool`
        self.num_workers = num_workers    # `int`
        self.pin_memory = pin_memory  # `bool`
        self.collate_fn = collate_fn
        self.block_dataset = BlockDataset(self.dataset, 
                                          batch_size=self.batch_size, 
                                          block_size=self.block_size)

    def __iter__(self):

        # define a regular `DataLoader` using the `BlockDataset`
        block_loader = torch.utils.data.DataLoader(self.block_dataset,
                                                   shuffle=self.shuffle,
                                                   num_workers=self.num_workers,
                                                   collate_fn=self.collate_fn)

        # define a condition for determining whether to drop the last block
        # this is done if the remainder block is very small (less than a tenth
        # the size of a normal block)
        condition = bool(
            int(self.block_dataset.__len__()/self.block_size) > 1 & 
            self.block_dataset.__len__()%self.block_size < self.block_size/10
        )

        # loop through and load BLOCKS of data every iteration
        for block in block_loader:
            tensors, mols = block
            tensors = [torch.squeeze(t) for t in tensors]
            mols = mols[0]

            # wrap each block in a `ShuffleBlock` dataset so that data can be
            # shuffled *within* blocks too
            batch_loader = torch.utils.data.DataLoader(dataset=ShuffleBlockWrapper(tensors, mols),
                                                       shuffle=self.shuffle,       
                                                       batch_size=self.batch_size,
                                                       num_workers=self.num_workers,
                                                       pin_memory=self.pin_memory,
                                                       drop_last=condition,
                                                       collate_fn=self.collate_fn)
            for batch in batch_loader:
                yield batch

    def __len__(self):
        # returns the number of graphs in the DataLoader
        n_blocks = len(self.dataset) // self.block_size
        n_rem = len(self.dataset) % self.block_size
        n_batch_per_block = self.__ceil__(self.block_size, self.batch_size)
        n_last = self.__ceil__(n_rem, self.batch_size)
        return n_batch_per_block * n_blocks + n_last

    def __ceil__(self, x, y):
        return (x + y - 1) // y
    
    
class BlockDataset(torch.utils.data.Dataset):
    """ Modified `Dataset` class which returns BLOCKS of data when 
    `__getitem__()` is called.
    """
    def __init__(self, dataset, batch_size=100, block_size=10000):
        assert block_size >= batch_size

        self.block_size = block_size  # `int`
        self.batch_size = batch_size  # `int`
        self.dataset = dataset        # `PickleDataset or MoleculeDataset`

    def __getitem__(self, idx):
        # returns a block of data from the dataset
        start = idx * self.block_size
        end = min((idx + 1) * self.block_size, len(self.dataset))
        return self.dataset[start:end]

    def __len__(self):
        # returns the number of blocks in the dataset
        return (len(self.dataset) + self.block_size - 1) // self.block_size


class ShuffleBlockWrapper:
    """ Extra class used to wrap a block of data, enabling data to get shuffled 
    *within* a block.
    """
    def __init__(self, tensors, mols):
        self.tensors = tensors
        self.mols = mols

    def __getitem__(self, idx):
        tensors = [d[idx] for d in self.tensors]
        
        return tensors[0], tensors[1], tensors[2], tensors[3], tensors[4], self.mols[idx]

    def __len__(self):
        return len(self.tensors[0])

    
