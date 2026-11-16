import pandas as pd
from pymilvus import DataType, MilvusClient
from sentence_transformers import SentenceTransformer

from src.matching.preprocess import clean

COLLECTION = "buy_products"
DIM = 384

buy = pd.read_csv("data/Buy.csv", encoding="latin-1")
texts = buy["name"].apply(clean).tolist()
ids = buy["id"].tolist()

model = SentenceTransformer("models/minilm-finetuned", device="cuda")
emb = model.encode(texts, batch_size=64, normalize_embeddings=True)

client = MilvusClient(uri="http://localhost:19530")
if client.has_collection(COLLECTION):
    client.drop_collection(COLLECTION)

schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=DIM)

index_params = client.prepare_index_params()
index_params.add_index(
    field_name="embedding",
    index_type="HNSW",
    metric_type="COSINE",
    params={"M": 16, "efConstruction": 200},
)
client.create_collection(COLLECTION, schema=schema, index_params=index_params)

client.insert(
    COLLECTION,
    data=[{"id": int(i), "embedding": e.tolist()} for i, e in zip(ids, emb)],
)
client.flush(COLLECTION)
print("Yüklenen kayıt:", client.get_collection_stats(COLLECTION)["row_count"])

# Hızlı doğrulama
q = model.encode([clean("Sony Turntable PSLX350H")], normalize_embeddings=True)
res = client.search(COLLECTION, data=q.tolist(), limit=3, output_fields=["id"])
names = dict(zip(buy["id"], buy["name"]))
for hit in res[0]:
    print(hit["id"], round(hit["distance"], 4), names[hit["id"]])