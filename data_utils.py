import torch

class BlockDataLoader(torch.utils.data.DataLoader):
    """ Main `DataLoader` class which has been modified so as to read training
    data from disk in blocks, as opposed to a single line at a time (as is done
    in the original `DataLoader` class).
    """
    def __init__(self, dataset, batch_size=100, block_size=10000,
                shuffle=True, n_workers=0, pin_memory=True):

        # define variables to be used throughout dataloading
        self.dataset = dataset        # `CODDataset` object
        self.batch_size = batch_size  # `int`
        self.block_size = block_size  # `int`
        self.shuffle = shuffle        # `bool`
        self.n_workers = n_workers    # `int`
        self.pin_memory = pin_memory  # `bool`
        self.block_dataset = BlockDataset(self.dataset, 
                                          batch_size=self.batch_size, 
                                          block_size=self.block_size)

    def __iter__(self):

        # define a regular `DataLoader` using the `BlockDataset`
        block_loader = torch.utils.data.DataLoader(self.block_dataset,
                                                   shuffle=self.shuffle,
                                                   num_workers=self.n_workers)

        # define a condition for determining whether to drop the last block
        # this is done if the remainder block is very small (less than a tenth
        # the size of a normal block)
        condition = bool(
            int(self.block_dataset.__len__()/self.block_size) > 1 & 
            self.block_dataset.__len__()%self.block_size < self.block_size/10
        )

        # loop through and load BLOCKS of data every iteration
        for block in block_loader:
            block = [torch.squeeze(b) for b in block]

            # wrap each block in a `ShuffleBlock` dataset so that data can be
            # shuffled *within* blocks too
            batch_loader = torch.utils.data.DataLoader(dataset=ShuffleBlockWrapper(block),
                                                       shuffle=self.shuffle,       
                                                       batch_size=self.batch_size,
                                                       num_workers=self.n_workers,
                                                       pin_memory=self.pin_memory,
                                                       drop_last=condition)
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
        self.dataset = dataset        # `CODDataset`

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
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return [d[idx] for d in self.data]

    def __len__(self):
        return len(self.data[0])


class CODDataset(torch.utils.data.Dataset):
    """ Reads and collects data from an pickle files with five "datasets"
    """
    def __init__(self, nodes, masks, edges, proximities, positions):

        self.nodes = nodes
        self.masks = masks
        self.edges = edges
        self.proximities = proximities
        self.positions = positions

        # get the number of elements in the dataset
        self.n_graphs = self.nodes.shape[0]

    def __getitem__(self, idx):
        # returns specific graph elements
        nodes_i = torch.from_numpy(self.nodes[idx]).type(torch.float32)
        masks_i = torch.from_numpy(self.masks[idx]).type(torch.float32)
        edges_i = torch.from_numpy(self.edges[idx]).type(torch.float32)
        proximities_i = torch.from_numpy(self.proximities[idx]).type(torch.float32)
        positions_i = torch.from_numpy(self.positions[idx]).type(torch.float32)

        return (nodes_i, masks_i, edges_i, proximities_i, positions_i)

    def __len__(self):
        # returns the number of graphs in the dataset
        return self.n_graphs