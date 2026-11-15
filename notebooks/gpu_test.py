import torch
from sentence_transformers import SentenceTransformer

print("CUDA var mı:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cuda")
emb = model.encode(["Apple iPhone 15 128GB", "iphone 15 128 gb apple"])
print("Embedding boyutu:", emb.shape)