import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn.utils.rnn import pad_sequence

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