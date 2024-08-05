import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
import numpy as np

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

def train(model, train_loader, val_loader, epochs, device, patience=5):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    criterion = nn.BCELoss()
    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            batch = batch[0].to(torch.long).to(device)  # Convert batch to long tensor and move to device
            outputs = model(batch)
            outputs = outputs.view(-1, outputs.size(-1))  # Flatten the output
            targets = nn.functional.one_hot(batch, num_classes=outputs.size(-1)).float().view(-1, outputs.size(-1)).to(device)  # One-hot encode targets
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        val_targets_list = []
        val_outputs_list = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch[0].to(torch.long).to(device)  # Convert batch to long tensor and move to device
                val_outputs = model(batch)
                val_outputs = val_outputs.view(-1, val_outputs.size(-1))  # Flatten the output
                val_targets = nn.functional.one_hot(batch, num_classes=val_outputs.size(-1)).float().view(-1, val_outputs.size(-1)).to(device)  # One-hot encode targets
                val_loss += criterion(val_outputs, val_targets).item()
                
                val_targets_list.extend(val_targets.cpu().numpy())
                val_outputs_list.extend(val_outputs.cpu().numpy())

        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        val_targets_array = np.array(val_targets_list)
        val_outputs_array = np.array(val_outputs_list) > 0.5  # Binarize the outputs
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), 'best_model.pth')
        elif epoch - best_epoch >= patience:
            print("Early stopping!")
            break

def tokenize_and_pad(texts, vocab, tokenizer, maxlen):
    tokens = [torch.tensor([vocab[token] for token in tokenizer(text)], dtype=torch.long) for text in texts]
    padded_tokens = pad_sequence(tokens, batch_first=True, padding_value=vocab['<pad>'])
    return padded_tokens[:, :maxlen]
