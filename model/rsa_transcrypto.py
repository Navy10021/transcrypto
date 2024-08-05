from model.transcrypto import *
from model.RSA import *
from model.transcrypto import *
import numpy as np

class TransCryptoRSA(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, ff_dim, num_layers, maxlen, public_key, private_key):
        super(TransCryptoRSA, self).__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.encoders = nn.ModuleList([TransformerEncoder(embed_dim, num_heads, ff_dim) for _ in range(num_layers)])
        self.decoders = nn.ModuleList([TransformerDecoder(embed_dim, num_heads, ff_dim) for _ in range(num_layers)])
        self.output_layer = nn.Linear(embed_dim, input_dim)
        self.maxlen = maxlen
        self.public_key = public_key
        self.private_key = private_key

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

    def encrypt(self, x):
        x = x.cpu().numpy().astype(np.int64)
        encrypted = [rsa_encrypt(self.public_key, item.tobytes()) for item in x]
        return encrypted

    def decrypt(self, x):
        decrypted = [np.frombuffer(rsa_decrypt(self.private_key, item), dtype=np.int64) for item in x]
        return torch.tensor(decrypted, dtype=torch.long).to(next(self.parameters()).device)