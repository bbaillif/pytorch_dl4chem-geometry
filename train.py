#!/usr/bin/env python
# coding: utf-8

import os

from test_tube import Experiment
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.utilities.seed import seed_everything
from argparse import ArgumentParser

from LitCoordAE import LitCoordAE
from data_utils import CODDataModule

def main(args):
    
    data_dir = '/home/bb596/rds/hpc-work/dl4chem/'
    dim_edge = 10
    
    seed = args.seed
    seed_everything(seed)

    dim_h = args.dim_h
    dim_f = args.dim_f
    batch_size = args.batch_size
    val_num_samples = args.val_num_samples
    alignment_type = args.alignment_type
    use_X = args.use_X 
    use_R = args.use_R 
    refine_steps = args.refine_steps
    refine_mom = args.refine_mom
    useFF = args.useFF
    w_reg = args.w_reg
    dataset = args.dataset
    

    if dataset == 'QM9' :
        mpnn_steps = 3
        n_max = 9
        dim_node = 22
    else :
        mpnn_steps = args.mpnn_steps
        n_max = 50
        dim_node = 35

    data_module = CODDataModule(dataset=dataset, 
                                data_dir=data_dir, 
                                batch_size=batch_size, 
                                val_num_sample=val_num_samples)

    if args.checkpoint_start is None :
        model = LitCoordAE(n_max=n_max,
                           dim_node=dim_node,
                           dim_edge=dim_edge, 
                           dim_h=dim_h, 
                           dim_f=dim_f, 
                           batch_size=batch_size, 
                           mpnn_steps=mpnn_steps, 
                           alignment_type=alignment_type, 
                           use_X=use_X, 
                           use_R=use_R,
                           useFF=useFF,
                           refine_steps=refine_steps, 
                           refine_mom=refine_mom, 
                           w_reg=w_reg,
                           val_num_samples=val_num_samples)

    else :
        model = LitCoordAE.load_from_checkpoint(args.checkpoint_start, strict=False,
                                                n_max=n_max,
                                                dim_node=dim_node, 
                                                dim_edge=dim_edge,
                                                dim_h=dim_h,
                                                dim_f=dim_f,
                                                batch_size=batch_size, 
                                                mpnn_steps=mpnn_steps, 
                                                alignment_type=alignment_type, 
                                                use_X=use_X, 
                                                use_R=use_R,
                                                useFF=useFF,
                                                refine_steps=refine_steps, 
                                                refine_mom=refine_mom, 
                                                w_reg=w_reg,
                                                val_num_samples=val_num_samples)

    tb_logger = pl_loggers.TensorBoardLogger('logs/')
    trainer = Trainer(logger=tb_logger, gpus=-1, num_nodes=2, accelerator='ddp')
    trainer.fit(model, data_module)


if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.realpath(__file__))
    parser = ArgumentParser(description='Train network')
    
    parser.add_argument('--dataset', type=str, default='QM9', choices=['COD', 'QM9', 'CSD'])
    parser.add_argument('--alignment_type', type=str, default='kabsch', choices=['default', 'linear', 'kabsch'])
    parser.add_argument('--seed', type=int, default=1334, help='random seed for experiments')
    parser.add_argument('--batch_size', type=int, default=20, help='batch size')
    parser.add_argument('--val_num_samples', type=int, default=10,
                        help='number of samples from prior used for validation')
    parser.add_argument('--use_X', action='store_true', help='use X as input for posterior of Z')
    parser.add_argument('--use_R', action='store_true', help='use R(X) as input for posterior of Z')
    parser.add_argument('--w_reg', type=float, default=1e-5, help='weight for conditional prior regularization')
    parser.add_argument('--refine_mom', type=float, default=0.99, help='momentum used for refinement')
    parser.add_argument('--refine_steps', type=int, default=0, help='number of refinement steps if requested')
    parser.add_argument('--useFF', action='store_true', help='use force field minimisation if testing')
    parser.add_argument('--dim_h', type=int, default=50, help='dimension of the hidden')
    parser.add_argument('--dim_f', type=int, default=100, help='dimension of the hidden')
    parser.add_argument('--mpnn_steps', type=int, default=5, help='number of mpnn steps')
    parser.add_argument('--checkpoint_start', type=str, default=None, help='checkpoint to the model on')

    args = parser.parse_args()

    main(args)