import torch
import torch.nn as nn
from torch.nn import functional as F
import math

"""
Code source: CS231N assignment https://cs231n.github.io/assignments2024/assignment3/.
This file defines layer types that are commonly used for transformers.
"""

class PositionalEncoding(nn.Module):
    """
    Encodes information about the positions of the tokens in the sequence. In
    this case, the layer has no learnable parameters, since it is a simple
    function of sines and cosines.
    """
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        """
        Construct the PositionalEncoding layer.

        Inputs:
         - embed_dim: the size of the embed dimension
         - dropout: the dropout value
         - max_len: the maximum possible length of the incoming sequence
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        assert embed_dim % 2 == 0
        # Create an array with a "batch dimension" of 1 (which will broadcast
        # across all examples in the batch).
        pe = torch.zeros(1, max_len, embed_dim)

        for i in range(max_len):
          for k in range(embed_dim // 2):
            pe[0, i, k * 2] = math.sin(i * 10000**(-k * 2 / embed_dim))
            pe[0, i, k * 2 + 1] = math.cos(i * 10000**(-k * 2 / embed_dim))

        # Make sure the positional encodings will be saved with the model
        # parameters (mostly for completeness).
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Element-wise add positional embeddings to the input sequence.

        Inputs:
         - x: the sequence fed to the positional encoder model, of shape
              (N, S, D), where N is the batch size, S is the sequence length and
              D is embed dim
        Returns:
         - output: the input sequence + positional encodings, of shape (N, S, D)
        """
        N, S, D = x.shape

        output = torch.empty((N, S, D))
        output = x + self.pe[0,:S,:]
        output = self.dropout(output)

        return output


class MultiHeadAttention(nn.Module):
    """
    A model layer which implements a simplified version of masked attention, as
    introduced by "Attention Is All You Need" (https://arxiv.org/abs/1706.03762).

    Usage:
      attn = MultiHeadAttention(embed_dim, num_heads=2)

      # self-attention
      data = torch.randn(batch_size, sequence_length, embed_dim)
      self_attn_output = attn(query=data, key=data, value=data)

      # attention using two inputs
      other_data = torch.randn(batch_size, sequence_length, embed_dim)
      attn_output = attn(query=data, key=other_data, value=other_data)
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Construct a new MultiHeadAttention layer.

        Inputs:
         - embed_dim: Dimension of the token embedding
         - num_heads: Number of attention heads
         - dropout: Dropout probability
        """
        super().__init__()
        assert embed_dim % num_heads == 0

        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        self.attn_drop = nn.Dropout(dropout)

        self.n_head = num_heads
        self.emd_dim = embed_dim
        self.head_dim = self.emd_dim // self.n_head

    def forward(self, query, key, value, attn_mask=None):
        """
        Calculate the masked attention output for the provided data, computing
        all attention heads in parallel.

        In the shape definitions below, N is the batch size, S is the source
        sequence length, T is the target sequence length, and E is the embedding
        dimension.

        Inputs:
        - query: Input data to be used as the query, of shape (N, S, E)
        - key: Input data to be used as the key, of shape (N, T, E)
        - value: Input data to be used as the value, of shape (N, T, E)
        - attn_mask: Array of shape (S, T) where mask[i,j] == 0 indicates token
          i in the source should not influence token j in the target.

        Returns:
        - output: Tensor of shape (N, S, E) giving the weighted combination of
          data in value according to the attention weights calculated using key
          and query.
        """
        # print('query shape', query.shape)
        # print('value shape', value.shape)
        N, S, E = query.shape
        N, T, E = value.shape
        output = torch.empty((N, S, E))

        Q = self.query(query) # shape (N, S, E)
        K = self.key(key) # shape (N, T, E)
        V = self.value(value) # shape (N, T, E)

        Q = Q.reshape(N, S, self.n_head, self.head_dim).transpose(1, 2) # shape (N, H, S, E/H)
        K = K.reshape(N, T, self.n_head, self.head_dim).transpose(1, 2) # shape (N, H, T, E/H)
        V = V.reshape(N, T, self.n_head, self.head_dim).transpose(1, 2) # shape (N, H, T, E/H)

        scores = Q @ K.transpose(2, 3) / math.sqrt(self.head_dim) # shape (N, H, S, T)
        if attn_mask is not None:
          scores = scores.masked_fill(attn_mask == 0, -float('inf'))
        attention_weights = F.softmax(scores , dim=-1) # shape (N, H, S, T)
        attention_weights = self.attn_drop(attention_weights) 
        output = attention_weights @ V # shape (N, H, S, E/H)
        output = output.transpose(1, 2) # shape (N, S, H, E/H)
        output = self.proj(output.reshape(N, S, E)) # shape: (N, S, E)

        return output


