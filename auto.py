### VALIDAR DATASET COMPLETO
import torch
import pickle
import clip
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# --------------------------------------------------
# DEFININDO INPUTS 
# --------------------------------------------------
def inputs():
    captions_input = 'captions.txt'
    name_img_input = 'name_img.pkl'
    image_to_legend_indices_input = 'image_to_legend_indices.pkl'
    all_legendas_input = 'all_legendas.pkl'
    modelo_openai_input = 'ViT-B/32'
    ja_feito_salvo_em_pickle = True

    modelo_openai_input_path_name = f"{modelo_openai_input.replace('/', '-')}"
    modelo_openai_input_path_name = f"mclip-b32"
    return captions_input, name_img_input, image_to_legend_indices_input, all_legendas_input, modelo_openai_input, modelo_openai_input_path_name, ja_feito_salvo_em_pickle

# --------------------------------------------------
# PATH LOCALIZACAO DATASET
# --------------------------------------------------
def define_path(captions_input, name_img_input, image_to_legend_indices_input, all_legendas_input, modelo_openai_input_path_name):
    project_folder = '/home/users/vcbenedet/FM30k/'
    tar_gz_path = f'{project_folder}flickr30k-images.tar.gz'
    captions_path = f'{project_folder}{captions_input}'
    path_image_folder = f'{project_folder}flickr30k-images'

    path_name_vector = f'{project_folder}{name_img_input}'
    path_image_to_legend_indices = f'{project_folder}{image_to_legend_indices_input}'
    path_all_legendas = f'{project_folder}{all_legendas_input}'
    path_folder_embeddings = f'{project_folder}Inferencias/'
    save_path_embeddings = os.path.join(path_folder_embeddings, f'Embeddings_{modelo_openai_input_path_name}')
    return project_folder, tar_gz_path, captions_path, path_image_folder, path_name_vector, path_image_to_legend_indices, path_all_legendas, save_path_embeddings

# --------------------------------------------------
# ORGANIZANDO IMAGENS E LEGENDAS
# --------------------------------------------------

def organiza_img_txt(captions_path, path_name_vector, path_image_to_legend_indices, path_all_legendas, ja_feito_salvo_em_pickle):
    print("Organizando imagens e legendas")

    ## -- FAZ OS PICKLES CASO NÃO TENHA ELES SALVOS (nome_img, all_legendas, image_to_legends) --
    if not ja_feito_salvo_em_pickle:
        images_names = []

        # Ler as legendas do arquivo
        with open(captions_path, 'r') as f:
            for idx, line in enumerate(f.readlines()):  # Use o índice do loop para o índice das legendas
                line = line.strip()  # Remover espaços em branco nas extremidades
                if not line:  # Ignorar linhas vazias
                    continue

                # Verificar se a linha contém duas tabulações para separar a imagem e a legenda
                parts = line.split('\t')

                # Separar a linha em partes (imagem, índice da legenda e o texto da legenda)
                image_name_index, caption_text = parts[0], parts[1]
                image_name_index_ = image_name_index.split('#')
                # Extrair nome da imagem e o índice da legenda
                image_name, index = image_name_index_[0], image_name_index_[1]

                images_names.append(image_name)

                image_index = images_names.index(image_name)

    if not ja_feito_salvo_em_pickle:
        # Lista para armazenar os arquivos únicos mantendo a ordem
        images_name_unique = []

        # Conjunto para verificar se o item já foi adicionado
        arquivos_vistos = set()

        # Iterando pelo vetor original
        for arquivo in images_names:
            if arquivo not in arquivos_vistos:
                images_name_unique.append(arquivo)
                arquivos_vistos.add(arquivo)

        with open(path_name_vector, 'wb') as file:
            pickle.dump(images_name_unique, file)

        print("Vetor path_name_vector salvo com sucesso usando pickle!")

    if not ja_feito_salvo_em_pickle:
        image_captions = {}
        image_to_legend_indices = {}  # Dicionário que mapeia índice da imagem para índices das legendas
        all_legendas = []
        images_names = images_names

        # Ler as legendas do arquivo
        with open(captions_path, 'r') as f:
            for idx, line in enumerate(f.readlines()):  # Use o índice do loop para o índice das legendas
                line = line.strip()  # Remover espaços em branco nas extremidades
                if not line:  # Ignorar linhas vazias
                    continue

                # Verificar se a linha contém duas tabulações para separar a imagem e a legenda
                parts = line.split('\t')

                # Separar a linha em partes (imagem, índice da legenda e o texto da legenda)
                image_name_index, caption_text = parts[0], parts[1]
                image_name_index_ = image_name_index.split('#')

                # Extrair nome da imagem e o índice da legenda
                image_name, index = image_name_index_[0], image_name_index_[1]
                index = int(index) - 1  # Ajusta para começar de 0

                # Adicionar o nome da imagem e a legenda nas listas
                all_legendas.append(caption_text)

                # Organizar as legendas em um dicionário, onde a chave é o nome da imagem
                if image_name not in image_captions:
                    image_captions[image_name] = {}
                image_captions[image_name][index] = caption_text

                # Encontre o índice da imagem na lista `images_names` carregada anteriormente
                image_index = images_names.index(image_name)

                # Se o índice da imagem ainda não estiver no dicionário `image_to_legend_indices`, adicione uma lista vazia
                if image_index not in image_to_legend_indices:
                    image_to_legend_indices[image_index] = []

                # Adicionar o índice da legenda (usando o índice do loop) à lista correspondente à imagem
                image_to_legend_indices[image_index].append(idx)

        with open(path_image_to_legend_indices, 'wb') as file:
                pickle.dump(image_to_legend_indices, file)
        print("Vetor path_image_to_legend_indices salvo com sucesso usando pickle!")
        

        with open(path_all_legendas, 'wb') as file:
                pickle.dump(all_legendas, file)
        print("Vetor path_all_legendas salvo com sucesso usando pickle!")
        

    ## -- LÊ OS PICKLES SALVOS (nome_img, all_legendas, image_to_legends) --

    with open(path_name_vector, 'rb') as file:
        images_names = pickle.load(file)

    with open(path_image_to_legend_indices, 'rb') as file:
        image_to_legend_indices = pickle.load(file)

    with open(path_all_legendas, 'rb') as file:
        all_legendas = pickle.load(file)

    return images_names, all_legendas, image_to_legend_indices

# --------------------------------------------------
# FUNCOES DE AVALIACAO
# --------------------------------------------------

def normalize_embeddings(embeddings):
    return embeddings / embeddings.norm(dim=1, keepdim=True)

def compute_metrics_at_k(save_path_embeddings, modelo_openai_input_path_name, image_to_cap_index, k=5, batch_size=512):
    # Load
    image_embeddings = torch.load(
        os.path.join(save_path_embeddings, f'image_embeddings_{modelo_openai_input_path_name}.pt')
    )
    text_embeddings = torch.load(
        os.path.join(save_path_embeddings, f'text_embeddings_{modelo_openai_input_path_name}.pt')
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_embeddings = image_embeddings.to(device)
    text_embeddings = text_embeddings.to(device)

    # Normalize
    image_embeddings = normalize_embeddings(image_embeddings)
    text_embeddings = normalize_embeddings(text_embeddings)

    num_images = image_embeddings.shape[0]
    topk_indices = []

    # Compute similarities in batches
    with torch.no_grad():
        for i in range(0, num_images, batch_size):
            batch = image_embeddings[i:i+batch_size]  # (B, D)
            sims = batch @ text_embeddings.T          # (B, N_texts)
            topk = torch.topk(sims, k=k, dim=1).indices  # (B, k)
            topk_indices.append(topk.cpu())

    topk_indices = torch.cat(topk_indices, dim=0).tolist()  # shape (num_images, k)

    # Ground truth sets
    ground_truth = []
    for i in range(num_images):
        caps = image_to_cap_index[i]
        if isinstance(caps, int):
            caps = [caps]
        ground_truth.append(set(caps))

    # Binary relevance
    y_true_binary = []
    y_pred_binary = []

    for i, gt_set in enumerate(ground_truth):
        preds = topk_indices[i]
        y_true_binary.extend([idx in gt_set for idx in preds])
        y_pred_binary.extend([True] * len(preds))

    # Metrics
    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

    # Accuracy@K
    correct = [
        any(idx in ground_truth[i] for idx in topk_indices[i])
        for i in range(num_images)
    ]
    accuracy = np.mean(correct)

    return {
        f'Precision@{k}': precision,
        f'Recall@{k}': recall,
        f'F1@{k}': f1,
        f'Accuracy@{k}': accuracy,
    }


# --------------------------------------------------
# CHAMADA DAS FUNÇÕES
# --------------------------------------------------
inicio_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"INICIO: {inicio_hora}")

captions_input, name_img_input, image_to_legend_indices_input, all_legendas_input, modelo_openai_input, modelo_openai_input_path_name, ja_feito_salvo_em_pickle  = inputs()
print("INPUTS")
print(f'Txt Captions: {captions_input}')
print(f'PKT Name_img: {name_img_input}')
print(f'PKT ITL: {image_to_legend_indices_input}')
print(f'PKT Legendas: {all_legendas_input}')
print(f'Modelo: {modelo_openai_input}')
print(f'Nome modelo salvar: {modelo_openai_input_path_name}')
print(f'Ja salvo pickle: {ja_feito_salvo_em_pickle}')

print("PATHS")
project_folder, tar_gz_path, captions_path, path_image_folder, path_name_vector, path_image_to_legend_indices, path_all_legendas, save_path_embeddings = define_path(captions_input, name_img_input, image_to_legend_indices_input, all_legendas_input, modelo_openai_input_path_name)

print(f'Path project_folder: {project_folder}')
print(f'Path tar_gz_path: {tar_gz_path}')
print(f'Path captions_path: {captions_path}')
print(f'Path path_image_folder: {path_image_folder}')
print(f'Path path_name_vector: {path_name_vector}')
print(f'Path path_image_to_legend_indices: {path_image_to_legend_indices}')
print(f'Path path_all_legendas: {path_all_legendas}')
print(f'Path save_path_embeddings: {save_path_embeddings}')

print("ESTRUTURAS")
images_names, all_legendas, image_to_legend_indices = organiza_img_txt(captions_path, path_name_vector, path_image_to_legend_indices, path_all_legendas, ja_feito_salvo_em_pickle)
print("Vetor images_names carregado com sucesso usando pickle!")
print(len(images_names))
print("Dicionário image_to_legend_indices carregado com sucesso usando pickle!")
print(len(image_to_legend_indices))
print("Vetor all_legendas carregado com sucesso usando pickle!")
print(len(all_legendas))

print("METRICAS")
for k in [1, 5, 10]:
    met = compute_metrics_at_k(save_path_embeddings, modelo_openai_input_path_name, image_to_legend_indices, k=k)
    print(k)
    print(met)

final_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"FINAL: {final_hora}")
