#!/usr/bin/env python
# coding: utf-8

import os
import torch

from test_tube import Experiment
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.utilities.seed import seed_everything
from argparse import ArgumentParser

from LitCoordAE import LitCoordAE
from data_utils import MoleculeDataModule

def main(args):
    
    seed_everything(args.seed)

    args.dim_edge = 10
    if args.dataset == 'QM9' :
        args.mpnn_steps = 3
        args.n_max = 9
        args.dim_node = 22
    elif args.dataset == 'COD' :
        args.mpnn_steps = 5
        args.n_max = 50
        args.dim_node = 35
    else : # CSD or will lead to error
        args.mpnn_steps = 5
        args.n_max = 50
        args.dim_node = 98
    

    data_module = MoleculeDataModule(dataset=args.dataset, 
                                data_dir=args.data_dir, 
                                batch_size=args.batch_size, 
                                val_num_sample=args.val_num_samples)

    if args.checkpoint_start is None :
        model = LitCoordAE(args)
    else :
        checkpoint = torch.load(args.checkpoint_start)
        hparams = checkpoint['hyper_parameters']
        model = LitCoordAE.load_from_checkpoint(args.checkpoint_start, hparams=hparams, strict=False)

    tb_logger = pl_loggers.TensorBoardLogger('logs/')
    trainer = Trainer.from_argparse_args(args, logger=tb_logger) #, precision=16
    trainer.fit(model, data_module)


if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.realpath(__file__))
    parser = ArgumentParser(description='Train network')
    
    parser.add_argument('--dataset', type=str, default='QM9', choices=['COD', 'QM9', 'CSD'])
    parser.add_argument('--data_dir', type=str, default='/home/bb596/rds/hpc-work/dl4chem/')
    parser.add_argument('--batch_size', type=int, default=20, help='batch size')
    parser.add_argument('--val_num_samples', type=int, default=10,
                        help='number of samples from prior used for validation')
    parser.add_argument('--checkpoint_start', type=str, default=None, help='checkpoint to start the model on')
    
    parser = LitCoordAE.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser) # add the gpus and num_nodes arguments for multi-nodes training on cluster

    args = parser.parse_args()

    main(args)