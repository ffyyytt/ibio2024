#!/bin/bash -l
# L'argument '-l' est indispensable pour bénéficier des directives de votre .bashrc
 
# On peut éventuellement placer ici les commentaires SBATCH permettant de définir les paramètres par défaut de lancement :
#SBATCH --gres gpu:1
#SBATCH --time 0-00:30:00
#SBATCH --cpus-per-task 8
#SBATCH --mem-per-cpu 3G
#SBATCH --mail-type FAIL,END

conda activate ffyytt
python3 main.py