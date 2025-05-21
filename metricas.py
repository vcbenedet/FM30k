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

# --------------------------------------------------
# DEFININDO INPUTS 
# --------------------------------------------------
def inputs():
    captions_input = 'captions.txt'
    name_img_input = 'name_img_1000.pkl'
    image_to_legend_indices_input = 'image_to_legend_indices_1000.pkl'
    all_legendas_input = 'all_legendas_1000.pkl'
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

def evaluate_fold_image_to_text(
    val_indices,
    image_embeddings_val,
    text_embeddings_val,
    all_legendas,
    image_to_legend_indices,
    val_legend_indices,
    k=5,
    batch_size=256
                  ):
    total_images = len(val_indices)

    correct_count = 0
    average_precisions = []
    precision_at_k_sum = 0.0
    recall_at_k_sum = 0.0
    reciprocal_ranks = []

    for batch_start in range(0, total_images, batch_size):
        batch = image_embeddings_val[batch_start: batch_start + batch_size]

        # Similaridade entre batch de imagens e todas as legendas de validação
        similarity = batch @ text_embeddings_val.T
        topk_indices = torch.topk(similarity, k=k, dim=1).indices.cpu().tolist()

        for j, indices in enumerate(topk_indices):
            image_idx = val_indices[batch_start + j]  # índice original da imagem
            ground_truth = set(image_to_legend_indices[image_idx])

            # Métricas
            # indices são os indices locais, 0 do text_embeddings pode ser a legenda global 250
            # (val_legend_indices mapeia os índices locais para globais)
            if any(val_legend_indices[idx] in ground_truth for idx in indices):
                correct_count += 1

            relevant_count = 0
            precision_sum = 0.0
            for rank, idx in enumerate(indices):
                legend_idx_global = val_legend_indices[idx]
                if legend_idx_global in ground_truth:
                    relevant_count += 1
                    precision_sum += relevant_count / (rank + 1)

            average_precision = precision_sum / min(len(ground_truth), k) if ground_truth else 0
            average_precisions.append(average_precision)

            precision_at_k = sum(1 for idx in indices if val_legend_indices[idx] in ground_truth) / k
            precision_at_k_sum += precision_at_k

            recall_at_k = sum(1 for idx in indices if val_legend_indices[idx] in ground_truth) / len(ground_truth)
            recall_at_k_sum += recall_at_k

            reciprocal_rank = 0.0
            for rank, idx in enumerate(indices):
                if val_legend_indices[idx] in ground_truth:
                    reciprocal_rank = 1.0 / (rank + 1)
                    break
            reciprocal_ranks.append(reciprocal_rank)

    # Resultados finais
    top5_accuracy = correct_count / total_images
    mean_average_precision = sum(average_precisions) / total_images
    precision_at_k = precision_at_k_sum / total_images
    recall_at_k = recall_at_k_sum / total_images
    mean_reciprocal_rank = sum(reciprocal_ranks) / total_images
    f1_score_at_k = (2 * precision_at_k * recall_at_k) / (precision_at_k + recall_at_k) if (precision_at_k + recall_at_k) > 0 else 0.0

    print(f"(Imagem ➔ Texto)")
    print(f"✅ Top-5 Accuracy: {top5_accuracy:.4f}")
    print(f"✅ mAP: {mean_average_precision:.4f}")
    print(f"✅ Precision@{k}: {precision_at_k:.4f}")
    print(f"✅ Recall@{k}: {recall_at_k:.4f}")
    print(f"✅ MRR: {mean_reciprocal_rank:.4f}")
    print(f"✅ F1-Score@{k}: {f1_score_at_k:.4f}")

    return {
        "top5_accuracy": top5_accuracy,
        "map": mean_average_precision,
        "precision@k": precision_at_k,
        "recall@k": recall_at_k,
        "mrr": mean_reciprocal_rank,
        "f1@k": f1_score_at_k,
    }

def evaluate_fold_text_to_image(
    val_image_indices,
    image_embeddings_val,
    text_embeddings_val,
    image_to_legend_indices,
    val_legend_indices,
    k=5,
    batch_size=256
):
    total_legendas = len(val_legend_indices)

    # Construir dicionário: legenda -> imagem
    legend_to_image = {}
    for image_idx, legend_indices in image_to_legend_indices.items():
        for legend_idx in legend_indices:
            legend_to_image[legend_idx] = image_idx

    correct_count = 0
    average_precisions = []
    precision_at_k_sum = 0.0
    recall_at_k_sum = 0.0
    reciprocal_ranks = []

    for batch_start in range(0, total_legendas, batch_size):
        batch = text_embeddings_val[batch_start: batch_start + batch_size]

        # Similaridade entre batch de legendas e todas as imagens
        similarity = batch @ image_embeddings_val.T
        topk_indices = torch.topk(similarity, k=k, dim=1).indices.cpu().tolist()

        for j, indices in enumerate(topk_indices):
            legend_idx_global = val_legend_indices[batch_start + j]

            # Pega a imagem correspondente a esta legenda
            ground_truth_image_idx = legend_to_image.get(legend_idx_global, None)
            if ground_truth_image_idx is None:
                continue  # Caso improvável: legenda sem imagem associada

            retrieved_images = [val_image_indices[idx] for idx in indices]

            # Métricas
            if ground_truth_image_idx in retrieved_images:
                correct_count += 1

            relevant_count = 0
            precision_sum = 0.0
            for rank, idx in enumerate(indices):
                image_idx_global = val_image_indices[idx]
                if image_idx_global == ground_truth_image_idx:
                    relevant_count += 1
                    precision_sum += relevant_count / (rank + 1)

            average_precision = precision_sum / min(1, k)  # Só uma imagem correta
            average_precisions.append(average_precision)

            precision_at_k = sum(1 for idx in indices if val_image_indices[idx] == ground_truth_image_idx) / k
            precision_at_k_sum += precision_at_k

            recall_at_k = sum(1 for idx in indices if val_image_indices[idx] == ground_truth_image_idx) / 1  # apenas 1 relevante
            recall_at_k_sum += recall_at_k

            reciprocal_rank = 0.0
            for rank, idx in enumerate(indices):
                if val_image_indices[idx] == ground_truth_image_idx:
                    reciprocal_rank = 1.0 / (rank + 1)
                    break
            reciprocal_ranks.append(reciprocal_rank)

    # Resultados finais
    topk_accuracy = correct_count / total_legendas
    mean_average_precision = sum(average_precisions) / total_legendas
    precision_at_k = precision_at_k_sum / total_legendas
    recall_at_k = recall_at_k_sum / total_legendas
    mean_reciprocal_rank = sum(reciprocal_ranks) / total_legendas
    f1_score_at_k = (2 * precision_at_k * recall_at_k) / (precision_at_k + recall_at_k) if (precision_at_k + recall_at_k) > 0 else 0.0

    print(f"(Texto ➔ Imagem)")
    print(f"✅ Top-{k} Accuracy: {topk_accuracy:.4f}")
    print(f"✅ mAP: {mean_average_precision:.4f}")
    print(f"✅ Precision@{k}: {precision_at_k:.4f}")
    print(f"✅ Recall@{k}: {recall_at_k:.4f}")
    print(f"✅ MRR: {mean_reciprocal_rank:.4f}")
    print(f"✅ F1-Score@{k}: {f1_score_at_k:.4f}")

    return {
        f"top{k}_accuracy": topk_accuracy,
        "map": mean_average_precision,
        f"precision@{k}": precision_at_k,
        f"recall@{k}": recall_at_k,
        "mrr": mean_reciprocal_rank,
        f"f1@{k}": f1_score_at_k,
    }

# --------------------------------------------------
# FUNCAO QUE CALCULA
# --------------------------------------------------

def metricas(images_names, all_legendas, image_to_legend_indices, save_path_embeddings, modelo_openai_input_path_name):
    val_indices = np.arange(len(images_names)) ## TODAS AS IMAGENS SÃO DE VALIDAÇÃO

    val_legend_indices = []
    for idx in val_indices:
        val_legend_indices.extend(image_to_legend_indices[idx])

    image_embeddings = torch.load(os.path.join(save_path_embeddings, f'image_embeddings_{modelo_openai_input_path_name}.pt'))
    text_embeddings = torch.load(os.path.join(save_path_embeddings, f'text_embeddings_{modelo_openai_input_path_name}.pt'))

    metricas_img = evaluate_fold_image_to_text(val_indices, image_embeddings, text_embeddings, all_legendas, image_to_legend_indices, val_legend_indices)
    metricas_text = evaluate_fold_text_to_image(val_indices, image_embeddings, text_embeddings, image_to_legend_indices, val_legend_indices)

    df_img = pd.DataFrame(list(metricas_img.items()), columns=['Metrica', 'Valor'])
    df_img.to_csv(f'{save_path_embeddings}/metricas_img_{modelo_openai_input_path_name}.csv', sep=';', decimal=',')
    df_text = pd.DataFrame(list(metricas_text.items()), columns=['Metrica', 'Valor'])
    df_text.to_csv(f'{save_path_embeddings}/metricas_text_{modelo_openai_input_path_name}.csv', sep=';', decimal=',')

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
metricas(images_names, all_legendas, image_to_legend_indices, save_path_embeddings, modelo_openai_input_path_name)

final_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"FINAL: {final_hora}")
