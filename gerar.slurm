#!/bin/bash
#SBATCH --job-name=clip_embed_completo
#SBATCH --partition=tupi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=log/%x_%j.out
#SBATCH --error=log/%x_%j.err

# Ativar ambiente virtual
source ~/FM30k/venv/bin/activate

# Executar o script
python3 gerar_embeddings_completo.py  
