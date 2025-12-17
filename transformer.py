import torch
import torch.nn as nn
import torch.optim as optim

class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        super(TransformerModel, self).__init__()
        # input embedding,
        # input_dim: size of the vocabulary
        # model_dim: dimension of the model
        self.embedding = nn.Embedding(input_dim, model_dim)

        # positional encoding
        # assuming max sequence length of 1000 for simplicity
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, model_dim))

        # transformer encoder layers
        self.transformer = nn.Transformer(d_model=model_dim,
                                          nhead=num_heads,
                                          num_encoder_layers=num_layers)
        # fully connected output layer
        self.fc = nn.Linear(model_dim, output_dim)

    def forward(self, src, tgt):
        src_seq_length, tgt_seq_length = src.size(1), tgt.size(1)  # [batch_size, src_seq_length, d_model]
        src = self.embedding(src) + self.positional_encoding[:, :src_seq_length, :]   # slice
        tgt = self.embedding(tgt) + self.positional_encoding[:, :tgt_seq_length, :]
        transformer_out = self.transformer(src, tgt)
        out = self.fc(transformer_out)
        return out

# Example usage:
input_dim = 10000  # vocabulary size
model_dim = 512    # dimension of the model
num_heads = 8      # number of attention heads
num_layers = 6     # number of transformer layers
output_dim = 10000 # vocabulary size for output

# create the model
model = TransformerModel(input_dim, model_dim, num_heads, num_layers, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# create random input and target sequences
src = torch.randint(0, input_dim, (10, 32))  # (batch_size, src_seq_length)
tgt = torch.randint(0, input_dim, (10, 32))  # (batch_size, tgt_seq_length)

# forward pass
output = model(src, tgt)

# compute loss
loss = criterion(output.view(-1, output_dim), tgt.view(-1))

# backward pass and optimization
optimizer.zero_grad()
loss.backward()
optimizer.step()


