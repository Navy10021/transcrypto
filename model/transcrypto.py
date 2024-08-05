import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerDecoder, self).__init__()
        self.attention1 = nn.MultiheadAttention(embed_dim, num_heads)
        self.attention2 = nn.MultiheadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.layernorm3 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)

    def forward(self, x, enc_output):
        attn1_output, _ = self.attention1(x, x, x)
        attn1_output = self.dropout1(attn1_output)
        out1 = self.layernorm1(x + attn1_output)
        attn2_output, _ = self.attention2(out1, enc_output, enc_output)
        attn2_output = self.dropout2(attn2_output)
        out2 = self.layernorm2(out1 + attn2_output)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        return self.layernorm3(out2 + ffn_output)

class TransCrypto(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, ff_dim, num_layers, maxlen):
        super(TransCrypto, self).__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.encoders = nn.ModuleList([TransformerEncoder(embed_dim, num_heads, ff_dim) for _ in range(num_layers)])
        self.decoders = nn.ModuleList([TransformerDecoder(embed_dim, num_heads, ff_dim) for _ in range(num_layers)])
        self.output_layer = nn.Linear(embed_dim, input_dim)
        self.maxlen = maxlen

    def forward(self, x):
        x = self.embedding(x) * (self.embed_dim ** 0.5)
        x = x.permute(1, 0, 2)  # Change shape to (seq_len, batch, embed_dim) for multihead attention
        for encoder in self.encoders:
            x = encoder(x)
        enc_output = x
        for decoder in self.decoders:
            x = decoder(x, enc_output)
        x = x.permute(1, 0, 2)  # Change shape back to (batch, seq_len, embed_dim)
        return torch.sigmoid(self.output_layer(x))