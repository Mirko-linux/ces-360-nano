# tokenizer/train_tokenizer.py

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os

# === MODIFICA QUI: percorso assoluto (più sicuro) ===
CORPUS_FILE = "C:/ces-360-nano/dataset/processed/corpus_leonese.txt"
TOKENIZER_SAVE_PATH = "tokenizer.json"

# Crea cartella se non esiste
os.makedirs(os.path.dirname(TOKENIZER_SAVE_PATH) if os.path.dirname(TOKENIZER_SAVE_PATH) else '.', exist_ok=True)

# === 1. Inizializza il tokenizer BPE (da zero) ===
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# === 2. Configura il trainer ===
trainer = BpeTrainer(
    vocab_size=50000,
    min_frequency=2,
    special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
)

# === 3. Addestra il tokenizer ===
print("🚀 Addestramento tokenizer in corso... (potrebbe richiedere 1-2 minuti)")

# 🔍 Debug: controlla se il file esiste
if not os.path.exists(CORPUS_FILE):
    raise FileNotFoundError(f"❌ File non trovato: {CORPUS_FILE}\nAssicurati di aver eseguito build_corpus.py")

tokenizer.train([CORPUS_FILE], trainer)

# === 4. Salva il tokenizer ===
tokenizer.save(TOKENIZER_SAVE_PATH)
print(f"✅ Tokenizer addestrato e salvato in: {TOKENIZER_SAVE_PATH}")
print(f"📊 Vocabolario: {tokenizer.get_vocab_size()} token")