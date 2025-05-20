import torch, numpy, PIL, platform, clip
print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("Numpy:", numpy.__version__)
print("PIL:", PIL.__version__)
print("CLIP:", clip.__file__)
print("OS:", platform.platform())
