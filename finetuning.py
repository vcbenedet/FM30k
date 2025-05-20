### GERAR EMBEDDINGS DATASET COMPLETO
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
    num_folds = 5
    seed = 42
    np.random.seed(seed)
    fold_id = 0
    epochs = 3
    base_finetuning = "ViT-B/32"
    nome_salvar = f"{base_finetuning.replace('/', '-')}_seed{seed}_{fold_id}"

    captions_input = 'captions.txt'
    name_img_input = 'name_img.pkl'
    image_to_legend_indices_input = 'image_to_legend_indices.pkl'
    all_legendas_input = 'all_legendas.pkl'
    ja_feito_salvo_em_pickle = True

    return captions_input, name_img_input, image_to_legend_indices_input, all_legendas_input, ja_feito_salvo_em_pickle, base_finetuning, num_folds, seed, fold_id, epochs, nome_salvar

# --------------------------------------------------
# PATH LOCALIZACAO DATASET
# --------------------------------------------------
def define_path(captions_input, name_img_input, image_to_legend_indices_input, all_legendas_input, nome_salvar):
    project_folder = '/home/users/vcbenedet/FM30k/'
    tar_gz_path = f'{project_folder}flickr30k-images.tar.gz'
    captions_path = f'{project_folder}{captions_input}'
    path_image_folder = f'{project_folder}flickr30k-images'

    path_name_vector = f'{project_folder}{name_img_input}'
    path_image_to_legend_indices = f'{project_folder}{image_to_legend_indices_input}'
    path_all_legendas = f'{project_folder}{all_legendas_input}'

    finetuning_folder = f'{project_folder}Finetuning/'
    finetuning_filename = f"clip_finetuned_{nome_salvar}.pth"
    finetuning_path = os.path.join(finetuning_folder, finetuning_filename)

    emb_fine_folder = f'{project_folder}Finetuning/Embeddings_Fine'
    emb_ori_folder = f'{project_folder}Finetuning/Embeddings_Original'


    return project_folder, tar_gz_path, captions_path, path_image_folder, path_name_vector, path_image_to_legend_indices, path_all_legendas, finetuning_path, emb_fine_folder, emb_ori_folder

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
# DIVISAO DATASET
# --------------------------------------------------
def generate_folds_pairs(images_names, num_folds=5, seed=42):
    """
    Gera folds de validação cruzada com embaralhamento controlado.
    Retorna uma lista de pares (train_indices, val_indices) como no KFold.

    Parâmetros:
    - images_names (list): Lista de nomes das imagens
    - num_folds (int): Número de folds (default: 5)
    - seed (int): Seed para embaralhamento reprodutível

    Retorna:
    - folds (list): Lista com tuplas (train_indices, val_indices)
    """
    # Gerar e embaralhar os índices
    image_indices = np.arange(len(images_names)) 
    rng = np.random.default_rng(seed)
    rng.shuffle(image_indices)

    # Dividir em folds
    folds_split = np.array_split(image_indices, num_folds)

    # Criar lista de pares (train, val)
    folds = []
    for i in range(num_folds):
        val_indices = folds_split[i]
        train_indices = np.concatenate([folds_split[j] for j in range(num_folds) if j != i])
        folds.append((train_indices.tolist(), val_indices.tolist()))

    return folds

def info_fold(folds, fold_id, images_names, all_legendas, image_to_legend_indices):
    train_indices, val_indices = folds[fold_id]

    # Obter imagens correspondentes
    train_images = [images_names[i] for i in train_indices]
    val_images = [images_names[i] for i in val_indices]

    # Obter as legendas correspondentes
    train_legends = [all_legendas[idx] for i in train_indices for idx in image_to_legend_indices[i]]
    val_legends = [all_legendas[idx] for i in val_indices for idx in image_to_legend_indices[i]]

    return train_indices, val_indices, train_images, val_images, train_legends, val_legends

# --------------------------------------------------
# FINETUNING
# --------------------------------------------------
def finetuning(images_names, image_to_legend_indices, base_finetuning, train_indices, train_legends, path_image_folder, epochs, finetuning_path):
    ## -- CARREGA E CONGELA O MODELO --
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(base_finetuning, device=device)

    # Congelar a parte visual
    for param in model.visual.parameters():
        param.requires_grad = False

    # Otimizador só para a parte textual
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

    loss_fn = torch.nn.CrossEntropyLoss()


    ## -- MONTA O DATASET --
    train_image_paths = []

    for i in train_indices:
        image_path = images_names[i]
        legend_indices = image_to_legend_indices[i]
        for _ in legend_indices:
            train_image_paths.append(image_path)

    class CLIPDataset(torch.utils.data.Dataset):
        def __init__(self, image_paths, legends, preprocess, image_dir):
            self.image_paths = image_paths
            self.legends = legends
            self.preprocess = preprocess
            self.image_dir = image_dir

        def __len__(self):
            return len(self.legends)

        def __getitem__(self, idx):
            image_filename = self.image_paths[idx]
            image_path = os.path.join(self.image_dir, image_filename)
            image = self.preprocess(Image.open(image_path).convert("RGB"))
            text = clip.tokenize([self.legends[idx]], truncate=True)[0].long()
            # text = self.legends[idx]
            return image, text
        
    train_dataset = CLIPDataset(train_image_paths, train_legends, preprocess, path_image_folder)

    ## -- DATALOADER --
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    ## -- EPOCAS - 1FOLD --
    model.train()
    model = model.float()
    torch.autograd.set_detect_anomaly(True)

    eps = 1e-8

    for epoch in range(epochs):
        total_loss = 0

        # Barra de progresso para a época
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")

        for batch_idx, (images, texts) in pbar:
            images = images.to(device).float()
            texts = texts.to(device)

            optimizer.zero_grad()

            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)

            image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + eps)
            text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + eps)

            logits_per_image = image_features @ text_features.t()
            logits_per_text = text_features @ image_features.t()

            labels = torch.arange(len(images), device=device)

            loss_img = loss_fn(logits_per_image, labels)
            loss_txt = loss_fn(logits_per_text, labels)
            loss = (loss_img + loss_txt) / 2

            # Verifica NaN ou Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print("❌ Detected NaN ou Inf na loss!")
                print("Loss img:", loss_img.item(), "Loss txt:", loss_txt.item())
                break

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # 📋 Logs de diagnóstico formatados
            diag_log = (
                f"Loss: {loss.item():.4f} | "
                f"I_feat μ: {image_features.mean():.4f}, σ: {image_features.std():.4f} | "
                f"T_feat μ: {text_features.mean():.4f}, σ: {text_features.std():.4f} | "
                f"Logits I→T: {logits_per_image.min():.2f}/{logits_per_image.max():.2f}"
            )

            # Atualiza a barra com info
            pbar.set_postfix_str(diag_log)

        print(f"\n Epoch {epoch+1}/{epochs} - Loss total: {total_loss:.4f}")

    

    # Salvar os pesos do modelo fine-tunado
    torch.save(model.state_dict(), finetuning_path)
    print(f"Modelo finetunado salvo em {finetuning_path}!")

# --------------------------------------------------
# EMBEDDINGS PARA VALIDACAO
# --------------------------------------------------
def embeddings_validacao(base_finetuning, finetuning_path, path_image_folder, val_images, val_legends, emb_fine_folder, emb_ori_folder, nome_salvar):
    ## -- DATASET E DATALOADER --
    class ImageOnlyDataset(Dataset):
        def __init__(self, image_paths, preprocess, image_dir):
            self.image_paths = image_paths  # Lista com nomes dos arquivos .jpg
            self.preprocess = preprocess
            self.image_dir = image_dir

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image_path = os.path.join(self.image_dir, self.image_paths[idx])
            image = Image.open(image_path).convert("RGB")
            return self.preprocess(image)

    class TextOnlyDataset(Dataset):
        def __init__(self, legends, tokenizer):
            self.legends = legends
            self.tokenizer = tokenizer  # clip.tokenize

        def __len__(self):
            return len(self.legends)

        def __getitem__(self, idx):
            tokens = self.tokenizer([self.legends[idx]], truncate=True)
            return tokens[0]
    
     ## -- FUNCAO QUE GERA OS EMBEDDINGS --
    
    def generate_embeddings(model, dataloader, tipo='image'):
        model.eval()
        all_embeddings = []

        with torch.no_grad():
            for batch in tqdm(dataloader):
                batch = batch.to(device)
                if tipo == 'image':
                    features = model.encode_image(batch)
                else:
                    features = model.encode_text(batch)

                features /= features.norm(dim=-1, keepdim=True)
                all_embeddings.append(features.cpu())

        return torch.cat(all_embeddings)
    
    ## -- CARREGANDO O MODELO FINETUNADO --
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(base_finetuning, device=device)

    model.load_state_dict(torch.load(finetuning_path, map_location=device))
    model.eval()  # Modo avaliação (desliga dropout, batchnorm, etc.)

    ## Finetuning
    image_dataset = ImageOnlyDataset(val_images, preprocess, path_image_folder)
    image_loader = DataLoader(image_dataset, batch_size=32, shuffle=False)

    text_dataset = TextOnlyDataset(val_legends, tokenizer=clip.tokenize)
    text_loader = DataLoader(text_dataset, batch_size=32, shuffle=False)

    image_embeddings = generate_embeddings(model, image_loader, tipo='image')
    text_embeddings = generate_embeddings(model, text_loader, tipo='text')
    
    os.makedirs(emb_fine_folder, exist_ok=True)
    # Salvar no disco
    torch.save(image_embeddings, os.path.join(emb_fine_folder, f'image_embeddings_val_fine_{nome_salvar}.pt'))
    torch.save(text_embeddings, os.path.join(emb_fine_folder, f'text_embeddings_val_fine_{nome_salvar}.pt'))
    print(f"Embeddings finetunado salvo em {emb_fine_folder}!")

    ## -- CARREGANDO O MODELO ORIGINAL --
    model_original, preprocess_original = clip.load(base_finetuning, device=device)

    ## Original
    image_dataset_original = ImageOnlyDataset(val_images, preprocess_original, path_image_folder)
    image_loader_original = DataLoader(image_dataset_original, batch_size=32, shuffle=False)

    text_dataset_original = TextOnlyDataset(val_legends, tokenizer=clip.tokenize)
    text_loader_original = DataLoader(text_dataset_original, batch_size=32, shuffle=False)

    image_embeddings_original = generate_embeddings(model_original, image_loader_original, tipo='image')
    text_embeddings_original = generate_embeddings(model_original, text_loader_original, tipo='text')
    
    os.makedirs(emb_ori_folder, exist_ok=True)
    # Salvar no disco
    torch.save(image_embeddings_original, os.path.join(emb_ori_folder, f'image_embeddings_val_ori_{nome_salvar}.pt'))
    torch.save(text_embeddings_original, os.path.join(emb_ori_folder, f'text_embeddings_val_ori_{nome_salvar}.pt'))
    print(f"Embeddings originais salvo em {emb_ori_folder}!")

# --------------------------------------------------
# VALIDACAO
# --------------------------------------------------
def validacao(val_indices, image_to_legend_indices, all_legendas, emb_fine_folder, emb_ori_folder, nome_salvar):
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
        print(f"Top-5 Accuracy: {top5_accuracy:.4f}")
        print(f"mAP: {mean_average_precision:.4f}")
        print(f"Precision@{k}: {precision_at_k:.4f}")
        print(f"Recall@{k}: {recall_at_k:.4f}")
        print(f"MRR: {mean_reciprocal_rank:.4f}")
        print(f"F1-Score@{k}: {f1_score_at_k:.4f}")

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
        print(f"Top-{k} Accuracy: {topk_accuracy:.4f}")
        print(f"mAP: {mean_average_precision:.4f}")
        print(f"Precision@{k}: {precision_at_k:.4f}")
        print(f"Recall@{k}: {recall_at_k:.4f}")
        print(f"MRR: {mean_reciprocal_rank:.4f}")
        print(f"F1-Score@{k}: {f1_score_at_k:.4f}")

        return {
            f"top{k}_accuracy": topk_accuracy,
            "map": mean_average_precision,
            f"precision@{k}": precision_at_k,
            f"recall@{k}": recall_at_k,
            "mrr": mean_reciprocal_rank,
            f"f1@{k}": f1_score_at_k,
        }

    val_legend_indices = []
    for idx in val_indices:
        val_legend_indices.extend(image_to_legend_indices[idx])


    ## Finetuning
    image_embeddings = torch.load(os.path.join(emb_fine_folder, f'image_embeddings_val_fine_{nome_salvar}.pt'))
    text_embeddings = torch.load(os.path.join(emb_fine_folder, f'text_embeddings_val_fine_{nome_salvar}.pt'))
    metricas_img = evaluate_fold_image_to_text(val_indices, image_embeddings, text_embeddings, all_legendas, image_to_legend_indices, val_legend_indices)
    metricas_text = evaluate_fold_text_to_image(val_indices, image_embeddings, text_embeddings, image_to_legend_indices, val_legend_indices)

    df_img = pd.DataFrame(list(metricas_img.items()), columns=['Métrica', 'Valor'])
    df_img.to_csv(os.path.join(emb_fine_folder, f'metricas_img_fine_{nome_salvar}.csv'), sep=';', decimal=',')
    df_text = pd.DataFrame(list(metricas_text.items()), columns=['Métrica', 'Valor'])
    df_text.to_csv(os.path.join(emb_fine_folder, f'metricas_text_fine_{nome_salvar}.csv'), sep=';', decimal=',')

    ## Original
    image_embeddings_original = torch.load(os.path.join(emb_ori_folder, f'image_embeddings_val_ori_{nome_salvar}.pt'))
    text_embeddings_original = torch.load( os.path.join(emb_ori_folder, f'text_embeddings_val_ori_{nome_salvar}.pt'))

    metricas_img = evaluate_fold_image_to_text(val_indices, image_embeddings_original, text_embeddings_original, all_legendas, image_to_legend_indices, val_legend_indices)
    metricas_text = evaluate_fold_text_to_image(val_indices, image_embeddings_original, text_embeddings_original, image_to_legend_indices, val_legend_indices)

    df_img = pd.DataFrame(list(metricas_img.items()), columns=['Métrica', 'Valor'])
    df_img.to_csv(os.path.join(emb_ori_folder, f'metricas_img_ori_{nome_salvar}.csv'), sep=';', decimal=',')
    df_text = pd.DataFrame(list(metricas_text.items()), columns=['Métrica', 'Valor'])
    df_text.to_csv(os.path.join(emb_ori_folder, f'metricas_text_ori_{nome_salvar}.csv'), sep=';', decimal=',')

# --------------------------------------------------
# CHAMADA DAS FUNÇÕES
# --------------------------------------------------
inicio_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"INICIO: {inicio_hora}")

captions_input, name_img_input, image_to_legend_indices_input, all_legendas_input, ja_feito_salvo_em_pickle, base_finetuning, num_folds, seed, fold_id, epochs, nome_salvar  = inputs()
print("INPUTS")
print(f'Txt Captions: {captions_input}')
print(f'PKT Name_img: {name_img_input}')
print(f'PKT ITL: {image_to_legend_indices_input}')
print(f'PKT Legendas: {all_legendas_input}')
print(f'base_finetuning: {base_finetuning}')
print(f'Ja salvo pickle: {ja_feito_salvo_em_pickle}')
print(f'N folds: {num_folds}')
print(f'seed: {seed}')
print(f'fold_id: {fold_id}')
print(f'epochs: {epochs}')
print(f'nome_salvar: {nome_salvar}')

print("PATHS")
project_folder, tar_gz_path, captions_path, path_image_folder, path_name_vector, path_image_to_legend_indices, path_all_legendas, finetuning_path, emb_fine_folder, emb_ori_folder = define_path(captions_input, name_img_input, image_to_legend_indices_input, all_legendas_input, nome_salvar)

print(f'Path project_folder: {project_folder}')
print(f'Path tar_gz_path: {tar_gz_path}')
print(f'Path captions_path: {captions_path}')
print(f'Path path_image_folder: {path_image_folder}')
print(f'Path path_name_vector: {path_name_vector}')
print(f'Path path_image_to_legend_indices: {path_image_to_legend_indices}')
print(f'Path path_all_legendas: {path_all_legendas}')
print(f'Path finetuning_path: {finetuning_path}')

print("ESTRUTURAS")
images_names, all_legendas, image_to_legend_indices = organiza_img_txt(captions_path, path_name_vector, path_image_to_legend_indices, path_all_legendas, ja_feito_salvo_em_pickle)
print("Vetor images_names carregado com sucesso usando pickle!")
print(len(images_names))
print("Dicionário image_to_legend_indices carregado com sucesso usando pickle!")
print(len(image_to_legend_indices))
print("Vetor all_legendas carregado com sucesso usando pickle!")
print(len(all_legendas))

print("FOLDS")
folds = generate_folds_pairs(images_names, num_folds, seed)
train_indices, val_indices, train_images, val_images, train_legends, val_legends = info_fold(folds, fold_id, images_names, all_legendas, image_to_legend_indices)

print("FINETUNING")
finetuning(images_names, image_to_legend_indices, base_finetuning, train_indices, train_legends, path_image_folder, epochs, finetuning_path)

print("EMBEDDINGS VALIDACAO")
embeddings_validacao(base_finetuning, finetuning_path, path_image_folder, val_images, val_legends, emb_fine_folder, emb_ori_folder, nome_salvar)

print("METRICAS")
validacao(val_indices, image_to_legend_indices, all_legendas, emb_fine_folder, emb_ori_folder, nome_salvar)

final_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"FINAL: {final_hora}")
