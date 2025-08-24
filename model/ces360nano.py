# model/ces360nano.py
import torch
import torch.nn as nn

class CES360Nano(nn.Module):
    def __init__(self,   vocab_size=50000,
    d_model=3072,      
    n_heads=24,        
    n_layers=16,       
    d_ff=12288,     
    max_len=1024
):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Embedding
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)

        # Layer comuni
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                nn.ModuleDict({
                    'self_attn': nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                    'mlp': nn.Sequential(
                        nn.Linear(d_model, d_ff),
                        nn.GELU(),
                        nn.Linear(d_ff, d_model)
                    ),
                    'ln1': nn.LayerNorm(d_model),
                    'ln2': nn.LayerNorm(d_model)
                })
            )

        # Head finale
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        device = input_ids.device
        B, T = input_ids.size()
        assert T <= self.max_len, f"Sequenza troppo lunga: {T} > {self.max_len}"

        # Posizioni
        pos = torch.arange(0, T, device=device).unsqueeze(0)

        # Embedding
        x = self.token_embed(input_ids) * (self.d_model ** 0.5) + self.pos_embed(pos)

        # Maschera per l'attenzione (non vede il futuro)
        attn_mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        attn_mask = attn_mask.masked_fill(attn_mask == 1, float('-inf'))

        # Passa attraverso i layer
        for layer in self.layers:
            # Self-attention
            x_norm = layer['ln1'](x)
            attn_output, _ = layer['self_attn'](
                x_norm, x_norm, x_norm,
                attn_mask=attn_mask
            )
            x = x + attn_output

            # Feed-forward
            x_norm = layer['ln2'](x)
            mlp_output = layer['mlp'](x_norm)
            x = x + mlp_output

        # Normalizzazione finale e testa
        x = nn.LayerNorm(self.d_model).to(device)(x)
        logits = self.lm_head(x)

        # Calcola loss
        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

        return logits, loss
