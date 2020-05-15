#!/bin/bash
#SBATCH --job-name=ElmoEMBDP70
#SBATCH --output=ElmoEMBDPGPU70
#SBATCH --mem=450G
#SBATCH --time=100:00:00#!/bin/bash
echo start
module load python/3.7.6-lihm
module load tensorflow2/py3.cuda10.0
pip install keras
echo installpackages
pip install numpy
pip install allennlp
pip install pandas
pip install csv
pip install pandas
pip install pickle
echo done
python ElmoScriptTogetEmbeddings.py

