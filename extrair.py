### EXTRAIR IMAGENS DO TAR GZ
import tarfile
import os

# Caminho da pasta no Google Drive
project_folder = '/home/users/vcbenedet/FM30k/'

# Caminhos dos arquivos
tar_gz_path = f'{project_folder}flickr30k-images.tar.gz'

# Extrair o arquivo .tar.gz
with tarfile.open(tar_gz_path, 'r:gz') as tar:
    tar.extractall(path=project_folder)

# Verificar os arquivos extraídos
extracted_files_path = os.path.join(project_folder, 'flickr30k-images')
extracted_files = os.listdir(extracted_files_path)
print(f"Quantidade arquivos extraídos: {len(extracted_files)}")  
print(f"Arquivos extraídos: {extracted_files[:5]}") # Mostrar os primeiros 5 arquivos extraídos
