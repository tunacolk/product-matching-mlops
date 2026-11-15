import random

from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

from src.matching.common import load_data, split_abt_ids

random.seed(42)

abt, buy, truth = load_data()
train_ids, val_ids, test_ids = split_abt_ids(abt, truth)

abt_text = dict(zip(abt["id"], abt["text"]))
buy_text = dict(zip(buy["id"], buy["text"]))

# Sadece TRAIN Abt ürünlerinin eşleşmelerinden çift üret
pairs = [
    {"anchor": abt_text[a], "positive": buy_text[b]}
    for a in train_ids
    for b in truth[a]
]
random.shuffle(pairs)
print("Eğitim çifti sayısı:", len(pairs))

model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

args = SentenceTransformerTrainingArguments(
    output_dir="models/minilm-finetuned",
    num_train_epochs=4,
    per_device_train_batch_size=32,
    learning_rate=2e-5,
    warmup_steps=0.1,
    fp16=True,
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    save_strategy="no",
    logging_steps=10,
    report_to="none",
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=Dataset.from_list(pairs),
    loss=MultipleNegativesRankingLoss(model),
)
trainer.train()
model.save("models/minilm-finetuned")