# pytorch_dl4chem-geometry
Pytorch implementation of dl4chem-geometry [Paper](https://www.nature.com/articles/s41598-019-56773-5) [Github](https://github.com/nyu-dl/dl4chem-geometry)
The goal is to generate molecular conformations, here 3D absolute coordinates.

A first implementation was built in Pytorch, then I switched on Pytorch Lightning (Lit prefix)
Use train.py to train the model, training metrics and checkpoint will be stored in the logs/ directory (check parser to see tunable arguments)
LitCoordAE.py contains the model and its submodules
data_utils.py contains the data structure to be used as input for the model (CODDataModule, CODDataset and some other extracted from [GraphInvent](https://github.com/MolecularAI/GraphINVENT/tree/master/graphinvent))
MSDScorer contains the various (R)MSD functions coded in the original dl4chem. The default one to be used is kabsch (which finds the reference conformation rotation to minimize RMSD between predicted and reference conformations)

Only 2 datasets (QM9 and COD) can be processed (use dataset preprocessing notebook). Beware as it requires at least 12 Go RAM to handle the edges data. Don't forget to change the used data directory.

First implementation (without Lightning) :
CoordAE.py
test.py (to evaluate the model, used as validation as well)
reproduce_Mansimov.py to run the model

If you don't want to use Pytorch Lightning (with the trainer), you could reuse the training_step / validation_step from the LitCoordAE.py in your main script, but you need to add the optimizer.zero_grad, loss.backward, optimizer.step to ensure the model training, and .to() calls handling the device to activate cuda training if necessary (as those steps are automatically handled by Lightning)

TODO for improvement :
- Better variable/class naming and harmonisation
- Implement test_step in LitCoordAE
- Add documentation / comment code

Note : to launch the code on a computing cluster with SLURM, please refer to Pytorch Lightning documentation on [Computing cluster](https://pytorch-lightning.readthedocs.io/en/stable/clouds/slurm.html) and [Multi GPU training](https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html), and don't forget the srun command in the sbatch script to run python.