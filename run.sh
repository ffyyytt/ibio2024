#SBATCH --gres gpu:1
#SBATCH --time 0-00:30:00
#SBATCH --cpus-per-task 8
#SBATCH --mem-per-cpu 3G
#SBATCH --mail-type FAIL,END

conda activate ffyytt
python3 main.py