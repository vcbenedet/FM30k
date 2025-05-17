#!/bin/bash
#SBATCH --job-name=exemplo1
#SBATCH --partition=tupi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:01:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

python3 hello.py
