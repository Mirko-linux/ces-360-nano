# utils/count_params.py

import sys
import os

# Aggiungi la cartella principale del progetto al percorso
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.ces360nano import CES360Nano

model = CES360Nano()

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"🔧 Parametri totali: {total_params:,}")
print(f"✅ Parametri addestrabili: {trainable_params:,}")
print(f"📊 ~{total_params / 1_000_000:.0f}M di parametri")