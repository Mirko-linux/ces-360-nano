# Configurazione globale di CES 360 Nano

# Percorsi
DATASET_RAW = "dataset/raw"
DATASET_PROCESSED = "dataset/processed/corpus_leonese.txt"
TOKENIZER_PATH = "tokenizer/tokenizer.json"
MODEL_SAVE_DIR = "checkpoints"

# Modello
VOCAB_SIZE = 50000
D_MODEL = 2048
N_LAYERS = 16
N_HEADS = 16
D_FF = 8192
MAX_LEN = 1024

# Training
BATCH_SIZE = 4
LEARNING_RATE = 3e-4
EPOCHS = 1
MAX_STEPS = 10000

# RAG
RAG_SEARCH_DEPTH = 5  # numero di risultati da DuckDuckGo
CACHE_FILE = "rag/cache.json"