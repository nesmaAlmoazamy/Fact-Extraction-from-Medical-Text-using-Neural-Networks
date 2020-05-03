#!/bin/bash
#SBATCH --job-name=DataAnnotation
#SBATCH --output=CharactesTaggingResults
#SBATCH --mem 300G
#SBATCH --time=100:00:00#!/bin/bash
#
echo start
#module load python-3.6.3
module load python/3.7.6-lihm
pip install numpy
pip install pandas
pip install csv

python characterTagDS.py

echo done
