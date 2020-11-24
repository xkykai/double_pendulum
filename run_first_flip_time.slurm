#!/bin/bash
#SBATCH -J first_flip_time
#SBATCH -o first_flip_time%j.out
#SBATCH -e first_flip_time%j.err
#SBATCH --mail-user=xinkai@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --exclusive

source /etc/profile
module load julia/1.5.2


julia first_flip_time_script.jl