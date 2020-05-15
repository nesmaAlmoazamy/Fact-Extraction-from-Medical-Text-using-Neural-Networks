#!/bin/bash
#SBATCH --job-name=Elmo5
#SBATCH --output=ElmoResults12
#SBATCH --mem=450G
#SBATCH -N 1
#SBATCH --time=100:00:00#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:3
echo start
module load python-3.6.3
#conda install -c estnltk estnltk


pip install anaconda
conda create -n myenv3.5  python=3.5
conda activate myenv3.5
conda install tensorflow-gpu=1.2 
echo install_TF
pip install tensorflow-gpu==1.2 h5py
echo LoadFile
#module load tensorflow2/py3.cuda10.0
module load tensorflow/py3.cuda90

echo runsetupfile
#python setup.py install
echo finishsetupfile
pip install bilm
python bin/dump_weights.py --save_dir='swb/checkpoint' --outfile 'swb/swb_weights.hdf5'

 
