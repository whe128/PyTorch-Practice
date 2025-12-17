import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

#============= Transformer model definition
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # dimension of each head

        # define linear layers for query, key, value and output
        self.W_q = nn.Linear(d_model, d_model)  # query
        self.W_k = nn.Linear(d_model, d_model)  # key
        self.W_v = nn.Linear(d_model, d_model)  # value
        self.W_o = nn.Linear(d_model, d_model)  # output

    def scaled_dot_product_attention(self, Q, K, V, mask = None):
        """
        Docstring for scaled_dot_product_attention

        :param Q: (batch_size, num_heads, seq_length, d_k)
        :param K: (batch_size, num_heads, seq_length, d_k)
        :param V: (batch_size, num_heads, seq_length, d_k)
        """

        # calculate attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # scaled

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # calculate attention weights
        attention_probs = torch.softmax(attention_scores, dim=-1)       # normalize the last dimension

        # weighted sum of values
        output = torch.matmul(attention_probs, V)  # (batch_size, num_heads, seq_length, d_k)
        return output

    def split_heads(self, x):
        """
        Docstring for split_heads
        input shape: (batch_size, seq_length, d_model)
        output shape: (batch_size, num_heads, seq_length, d_k)
        """
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        """
        Docstring for combine_heads
        input shape: (batch_size, num_heads, seq_length, d_k)
        output shape: (batch_size, seq_length, d_model)
        """
        batch_size, num_heads, seq_length, d_k = x.size()
        # contiguous to ensure memory layout
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask = None):
        """
        forward propagation
        input shape(Q/K/V): (batch_size, seq_length, d_model)
        output shape: (batch_size, seq_length, d_model)
        """

        # linear transformation and split into heads
        Q = self.split_heads(self.W_q(Q))  # (batch_size, num_heads, seq_length, d_k)
        K = self.split_heads(self.W_k(K))  # (batch_size, num_heads, seq_length, d_k)
        V = self.split_heads(self.W_v(V))  # (batch_size, num_heads, seq_length, d_k)

        # calculate attention
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)  # (batch_size, num_heads, seq_length, d_k)

        # combine heads
        output = self.W_o(self.combine_heads(attention_output))  # (batch_size, seq_length, d_model)
        return output

# ============= Position-wise Feed-Forward Network
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        # first linear layer
        self.fc1 = nn.Linear(d_model, d_ff)
        # second linear layer
        self.fc2 = nn.Linear(d_ff, d_model)
        # activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch_size, seq_length, d_model)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# ============= Transformer Encoder Layer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init()
        pe = torch.zeros(max_seq_length, d_model)   # initialize positional encoding matrix
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        self.registers_buffer('pe', pe.unsqueeze(0))  # shape (1, max_seq_length, d_model)

    def forward(self, x):
        # add positional encoding to input embeddings
        return x + self.pe[:, :x.size(1)]

# ============= build encoder layer
class EncoderLayer(nn.Module):
    def __init__(self, d_modelm, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_modelm, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_modelm, d_ff)
        self.norm1 = nn.LayerNorm(d_modelm)
        self.norm2 = nn.LayerNorm(d_modelm)
        self.dropout = nn.Dropout(dropout)

    # forward propagation
    def forward(self, x, mask=None):
        # self-attention sublayer
        attention_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_output))

        # feed-forward sublayer
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x

# ============= build transformer encoder
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # self-attention
        attention_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attention_output))

        # cross-attention
        cross_attention_output = self.cross_attention(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attention_output))

        # feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

# ============= build transformer model
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        # build encoder layers
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        # final linear layer
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        # create src mask to hide filled characters
        # (batch_size, 1, 1, seq_length)
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

        # create tgt mask to hide future tokens in target sequence
        # (batch_size, 1, seq_length, 1)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)

        # create upper triangular matrix mask, avoid see the future information
        nopeak_mask = (1 - torch.triu(torch.ones((1, seq_length, seq_length), device=tgt.device), diagonal=1)).bool()

        # combine tgt_mask and nopeak_mask
        tgt_mask = tgt_mask & nopeak_mask

        return src_mask, tgt_mask

    def forward(self, src, tgt):
        # generate masks
        src_mask, tgt_mask = self.generate_mask(src, tgt)

        # encoder embedding + positional encoding
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        # decoder embedding + positional encoding
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        # final linear layer
        output = self.fc_out(dec_output)
        return output

# train model
# hyperparameters
src_vocab_size = 5000
tgt_vocab_size = 5000
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 100
dropout = 0.1

# initialize model
transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

# generate random input and target sequences
src_data = torch.randint(0, src_vocab_size, (32, 20))  # (batch_size, src_seq_length)
tgt_data = torch.randint(0, tgt_vocab_size, (32, 20))  # (batch_size, tgt_seq_length)

# define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(transformer.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)

# train
transformer.train()
for epoch in range(100):
    optimizer.zero_grad()

    # remove the last token, use for predict the next token
    output = transformer(src_data, tgt_data[:, :-1])  # input to decoder excludes last token

    # compute loss
    loss = criterion(
        output.contiguous().view(-1, tgt_vocab_size),
        tgt_data[:, 1:].contiguous().view(-1)  # target excludes first token
    )

    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

# ============= evaluate model
transformer.eval()
val_src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))
val_tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))

with torch.no_grad():
    val_output =  transformer(val_src_data, val_tgt_data[:, :-1])
    val_loss = criterion(
        val_output.contiguous().view(-1, tgt_vocab_size),
        val_tgt_data[:, 1:].contiguous().view(-1)
    )
    print(f'Validation Loss: {val_loss.item():.4f}')
