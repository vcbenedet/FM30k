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
from transformers import AutoTokenizer
from multilingual_clip import pt_multilingual_clip

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

    modelo_multi_input_path_name = "mclip-b32"
    modelo_multi_input_path_name = f"{modelo_multi_input_path_name}"

    return captions_input, name_img_input, image_to_legend_indices_input, all_legendas_input, modelo_openai_input, modelo_multi_input_path_name, ja_feito_salvo_em_pickle


# --------------------------------------------------
# PATH LOCALIZACAO DATASET
# --------------------------------------------------
def define_path(captions_input, name_img_input, image_to_legend_indices_input, all_legendas_input, modelo_multi_input_path_name):
    project_folder = '/home/users/vcbenedet/FM30k/'
    tar_gz_path = f'{project_folder}flickr30k-images.tar.gz'
    captions_path = f'{project_folder}{captions_input}'
    path_image_folder = f'{project_folder}flickr30k-images'

    path_name_vector = f'{project_folder}{name_img_input}'
    path_image_to_legend_indices = f'{project_folder}{image_to_legend_indices_input}'
    path_all_legendas = f'{project_folder}{all_legendas_input}'
    path_folder_embeddings = f'{project_folder}Inferencias/'
    save_path_embeddings = os.path.join(path_folder_embeddings, f'Embeddings_{modelo_multi_input_path_name}')
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
# GERAÇÃO DOS EMBEDDINGS DATASET COMPLETO
# --------------------------------------------------

def geracao_embeddings_multilingue(images_names, all_legendas, image_to_legend_indices, path_image_folder, modelo_openai_input, save_path_embeddings, modelo_multi_input_path_name):
    print("Geração dos Embeddings com M-CLIP (Multilingual CLIP)")

    # Dispositivo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device usado: {device}")

    # --- MODELO DE TEXTO M-CLIP ---
    device_m = "cpu"
    model_name = 'M-CLIP/XLM-Roberta-Large-Vit-B-32'
    model_text = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name).to(device_m)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # --- MODELO DE IMAGEM OPENAI CLIP ---
    model_image, preprocess_openai = clip.load(modelo_openai_input, device=device)

    # --- DATASETS E LOADERS ---
    class ImageOnlyDataset(Dataset):
        def __init__(self, image_paths, preprocess, image_dir):
            self.image_paths = image_paths
            self.preprocess = preprocess
            self.image_dir = image_dir

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image_path = os.path.join(self.image_dir, self.image_paths[idx])
            image = Image.open(image_path).convert("RGB")
            return self.preprocess(image)

    class TextOnlyDataset(Dataset):
        def __init__(self, legends):
            self.legends = legends

        def __len__(self):
            return len(self.legends)

        def __getitem__(self, idx):
            return self.legends[idx]

    # Define dados
    val_indices = np.arange(len(images_names))
    val_images = [images_names[i] for i in val_indices]
    val_legends = [all_legendas[idx] for i in val_indices for idx in image_to_legend_indices[i]]

    image_dataset = ImageOnlyDataset(val_images, preprocess_openai, path_image_folder)
    text_dataset = TextOnlyDataset(val_legends)

    image_loader = DataLoader(image_dataset, batch_size=32, shuffle=False)
    text_loader = DataLoader(text_dataset, batch_size=32, shuffle=False)

    # --- FUNÇÕES DE EMBEDDINGS ---
    def generate_image_embeddings(model, dataloader):
        model.eval()
        all_embeddings = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Imagem"):
                batch = batch.to(device)
                features = model.encode_image(batch)
                features = features / features.norm(dim=-1, keepdim=True)
                all_embeddings.append(features.cpu())
        return torch.cat(all_embeddings)

    # def generate_text_embeddings(model, tokenizer, dataloader):
    #     all_embeddings = []
    #     for batch in tqdm(dataloader, desc="Texto"):
    #         with torch.no_grad():
    #             # Tokeniza e move para o device (cuda ou cpu)
    #             tokenized = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
    #             tokenized = {k: v.to(device) for k, v in tokenized.items()}  # <- ESSENCIAL

    #             # Passa pela transformer do M-CLIP
    #             output = model.transformer(**tokenized)[0]  # shape: (batch_size, seq_len, hidden_dim)
    #             features = output[:, 0, :]  # CLS token

    #             # Normaliza e move para CPU
    #             features = features / features.norm(dim=-1, keepdim=True)
    #             all_embeddings.append(features.cpu())

    #     return torch.cat(all_embeddings)

    def generate_text_embeddings(model, tokenizer, dataloader):
        all_embeddings = []
        for batch in tqdm(dataloader, desc="Texto"):
            with torch.no_grad():
                embeddings = model(batch, tokenizer).to(device_m)  # já cuida de tudo internamente
                all_embeddings.append(embeddings.cpu())
        return torch.cat(all_embeddings)

    # def generate_text_embeddings(model, tokenizer, dataloader):
    #     all_embeddings = []
    #     model.eval()
    #     for batch in tqdm(dataloader, desc="Texto"):
    #         with torch.no_grad():
    #             tokens = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
    #             embeddings = model(tokens)  # model chama internamente o forward
    #             embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    #             all_embeddings.append(embeddings.cpu())
    #     return torch.cat(all_embeddings)

    # def generate_text_embeddings(model, tokenizer, dataloader):
    #     all_embeddings = []
    #     for batch in tqdm(dataloader, desc="Texto"):
    #         with torch.no_grad():
    #             # Tokeniza e move para o device
    #             tokens = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
    #             embeddings = model.transformer(**tokens)[0][:, 0]  # pega o CLS
    #             embeddings = model.text_projection(embeddings)
    #             embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                
    #             all_embeddings.append(embeddings.cpu())
    #     return torch.cat(all_embeddings)

    # def generate_text_embeddings(model, tokenizer, dataloader):
    #     all_embeddings = []
    #     for batch in tqdm(dataloader, desc="Texto"):
    #         with torch.no_grad():
    #             # Tokeniza o batch de textos
    #             inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
    #             # Extrai embeddings de texto
    #             embeddings = model.forward_text(**inputs)
    #             all_embeddings.append(embeddings.cpu())
    #     return torch.cat(all_embeddings)

    # --- GERAÇÃO ---
    text_emb = generate_text_embeddings(model_text, tokenizer, text_loader)
    image_emb = generate_image_embeddings(model_image, image_loader)
    print("Shape")
    print(text_emb.shape)
    os.makedirs(save_path_embeddings, exist_ok=True)

    torch.save(image_emb, os.path.join(save_path_embeddings, f"image_embeddings_{modelo_multi_input_path_name}.pt"))
    print(f"Embeddings de imagem salvos como image_embeddings_{modelo_multi_input_path_name}.pt")

    torch.save(text_emb, os.path.join(save_path_embeddings, f"text_embeddings_{modelo_multi_input_path_name}.pt"))
    print(f"Embeddings de texto salvos como text_embeddings_{modelo_multi_input_path_name}.pt")
# --------------------------------------------------
# CHAMADA DAS FUNÇÕES
# --------------------------------------------------
inicio_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"INICIO: {inicio_hora}")

captions_input, name_img_input, image_to_legend_indices_input, all_legendas_input, modelo_openai_input, modelo_multi_input_path_name, ja_feito_salvo_em_pickle  = inputs()
print("INPUTS")
print(f'Txt Captions: {captions_input}')
print(f'PKT Name_img: {name_img_input}')
print(f'PKT ITL: {image_to_legend_indices_input}')
print(f'PKT Legendas: {all_legendas_input}')
print(f'Modelo: {modelo_openai_input}')
print(f'Nome modelo salvar: {modelo_multi_input_path_name}')
print(f'Ja salvo pickle: {ja_feito_salvo_em_pickle}')

print("PATHS")
project_folder, tar_gz_path, captions_path, path_image_folder, path_name_vector, path_image_to_legend_indices, path_all_legendas, save_path_embeddings = define_path(captions_input, name_img_input, image_to_legend_indices_input, all_legendas_input, modelo_multi_input_path_name)

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

print("GERACAO")
geracao_embeddings_multilingue(images_names, all_legendas, image_to_legend_indices, path_image_folder, modelo_openai_input, save_path_embeddings, modelo_multi_input_path_name)
final_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"FINAL: {final_hora}")
