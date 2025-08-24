# model/trainer.py

import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from model.ces360nano import CES360Nano
import os
import time

# --- CONFIGURAZIONE ---
CORPUS_PATH = "../dataset/processed/corpus_leonese.txt"
TOKENIZER_PATH = "../tokenizer/tokenizer.json"
CHECKPOINT_DIR = "/content/drive/MyDrive/ces-360-nano/checkpoints"  # Cambia se in locale
BATCH_SIZE = 2
MAX_LEN = 1024
LEARNING_RATE = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_EVERY = 50    # Salva ogni N step
PRINT_EVERY = 10   # Stampa loss ogni N step

# Crea cartella checkpoint
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- DATASET ---
class LeoneseDataset(Dataset):
    def __init__(self, tokens, max_len=1024):
        self.tokens = tokens
        self.max_len = max_len

    def __len__(self):
        return max(1, len(self.tokens) // self.max_len)

    def __getitem__(self, idx):
        start = idx * self.max_len
        end = start + self.max_len
        if end + 1 > len(self.tokens):
            # Se siamo alla fine, ricomincia (piccolo hack per non rompere)
            start = max(0, len(self.tokens) - self.max_len - 1)
            end = start + self.max_len
        x = self.tokens[start:end]
        y = self.tokens[start+1:end+1]
        return torch.tensor(x), torch.tensor(y)

# --- CARICA DATI ---
print("📚 Caricamento corpus e tokenizer...")
try:
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
except Exception as e:
    raise FileNotFoundError(f"❌ Tokenizer non trovato in: {TOKENIZER_PATH}\n{e}")

try:
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"✅ Corpus caricato: {len(text):,} caratteri")
except Exception as e:
    raise FileNotFoundError(f"❌ Corpus non trovato in: {CORPUS_PATH}\n{e}")

encoded = tokenizer.encode(text)
tokens = encoded.ids
print(f"🔢 Token totali: {len(tokens):,}")

dataset = LeoneseDataset(tokens, MAX_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- MODELLO ---
print(f"🧠 Creazione modello CES 360 Nano ({DEVICE})...")
vocab_size = tokenizer.get_vocab_size()

model = CES360Nano(
    vocab_size=vocab_size,
    d_model=3072,      # ~2B parametri
    n_heads=24,
    n_layers=16,
    d_ff=12288,
    max_len=MAX_LEN
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# --- RIPRESA DA CHECKPOINT (se esiste) ---
start_step = 0
checkpoint_files = []
try:
    checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("ces360nano_step_") and f.endswith(".pt")]
except Exception as e:
    print(f"[!] Cartella checkpoint non leggibile: {e}")

if checkpoint_files:
    latest_checkpoint = sorted(
        checkpoint_files,
        key=lambda x: int(x.split("_")[-1].split(".")[0])
    )[-1]
    checkpoint_path = os.path.join(CHECKPOINT_DIR, latest_checkpoint)
    
    print(f"🔁 Ripresa da checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Carica stato modello e ottimizzatore
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    start_step = ckpt['step']
    print(f"🟢 Ripresa dallo step {start_step}, loss precedente: {ckpt['loss']:.4f}")
else:
    print("🆕 Nessun checkpoint trovato. Inizio addestramento da zero.")

# --- TRAINING LOOP ---
print("🔥 Inizio addestramento...")

model.train()
step = start_step
running_loss = 0.0

try:
    for epoch in range(100):  # Tante epoche, ma puoi fermarti quando vuoi
        print(f"🔄 Epoca {epoch + 1}")
        for batch in dataloader:
            if step < start_step:
                step += 1
                continue

            input_ids, targets = batch
            input_ids, targets = input_ids.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            try:
                logits, loss = model(input_ids, targets)
                loss.backward()
                optimizer.step()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("❌ Errore: out of memory. Riduci batch_size o usa una GPU più potente.")
                    torch.cuda.empty_cache()
                else:
                    raise e

            running_loss += loss.item()

            if step % PRINT_EVERY == 0:
                avg_loss = running_loss / PRINT_EVERY
                print(f"Step {step:5d} | Loss: {avg_loss:.4f} | LR: {LEARNING_RATE}")
                running_loss = 0.0

            if step % SAVE_EVERY == 0 and step > 0:
                checkpoint_path = f"{CHECKPOINT_DIR}/ces360nano_step_{step}.pt"
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item()
                }, checkpoint_path)
                print(f"💾 Checkpoint salvato: {checkpoint_path}")

            step += 1

            # Limita step per debug (rimuovi in produzione)
            # if step > start_step + 200: break

except KeyboardInterrupt:
    print("\n⏸️ Training interrotto manualmente. Salvataggio checkpoint...")
except Exception as e:
    print(f"\n❌ Errore durante il training: {e}")

# Salva all'uscita
final_path = f"{CHECKPOINT_DIR}/ces360nano_step_{step}_FINAL.pt"
torch.save({
    'step': step,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss.item()
}, final_path)
print(f"✅ Ultimo checkpoint salvato: {final_path}")

print("🎉 Addestramento completato (o interrotto). Riprendi quando vuoi.")