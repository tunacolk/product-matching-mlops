import pandas as pd

abt = pd.read_csv("data/Abt.csv", encoding="latin-1")
buy = pd.read_csv("data/Buy.csv", encoding="latin-1")
gt = pd.read_csv("data/abt_buy_perfectMapping.csv")

print("Abt:", abt.shape, "| Buy:", buy.shape, "| Gerçek eşleşme:", gt.shape)
print("\nAbt örnek:\n", abt.head(3)[["id", "name"]])
print("\nBuy örnek:\n", buy.head(3)[["id", "name"]])
print("\nEşleşme örnek:\n", gt.head(3))