#!/bin/bash
#SBATCH -J first_flip_time
#SBATCH -o logs/first_flip_time%j.out
#SBATCH -e logs/first_flip_time%j.err
#SBATCH -n 40
#SBATCH -N 1

source /etc/profile
module load julia/1.5.2

julia first_flip_time_script.jl