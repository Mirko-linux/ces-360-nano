# model/test_model.py

import torch
from ces360nano import CES360Nano

# 1. Crea il modello
print("🧠 Creazione di CES 360 Nano...")
model = CES360Nano(
    vocab_size=50000,
    d_model=2048,
    n_heads=16,
    n_layers=16,
    d_ff=8192,
    max_len=1024
)

# 2. Crea input di test (1 sequenza di 100 token)
print("🧪 Test input...")
input_ids = torch.randint(0, 50000, (1, 100))  # (batch_size=1, seq_len=100)
print(f"Input shape: {input_ids.shape}")

# 3. Esegui il forward
print("🔁 Esecuzione forward...")
with torch.no_grad():  # no gradient per test
    logits, loss = model(input_ids, targets=input_ids)

print(f"✅ Logits shape: {logits.shape} → (1, 100, 50000)")
print(f"✅ Loss: {loss.item():.4f}")
print("🎉 CES 360 Nano funziona! Il modello è pronto per l'addestramento.")