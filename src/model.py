import torch
from torch import nn, Tensor
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float=0.1, max_len: int=50000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        
        a = torch.sin(position * div_term)
        b = torch.cos(position * div_term)
        pe[:, 0, 0::2] = a
        pe[:, 0, 1::2] = b[:, :d_model-a.shape[1]]
        self.register_buffer('pe', pe)

    def get_pos_enc(self, shape) -> Tensor:

        return self.pe[:shape]


class Transformer(nn.Module):

    def __init__(self, d_model, n_enc1, n_enc2, n_dims, seq_len, n_head, n_classes, device, dim_ff, dropout_ff, concat_mode, select_mode, embedding_mode) -> None:
        super().__init__()

        self.d_model = d_model
        self.n_enc1 = n_enc1
        self.n_enc2 = n_enc2
        self.n_dims = n_dims
        self.seq_len = seq_len
        self.dim_ff = dim_ff
        self.dropout_ff = dropout_ff
        self.concat_mode = concat_mode
        self.select_mode = select_mode
        self.embedding_mode = embedding_mode

        assert self.seq_len % self.d_model == 0
        self.n_patch = int(self.seq_len / self.d_model)

        self.n_head = n_head
        self.n_classes = n_classes

        if self.embedding_mode:
            self.emb = nn.Linear(self.d_model, self.d_model)

        self.pos_enc = PositionalEncoding(self.d_model)
        self.pos_v = self.pos_enc.get_pos_enc(self.n_patch).reshape(1, 1, self.n_patch, self.d_model).to(device)
        
        self.layers = nn.ModuleDict()

        for i in range(self.n_enc1):
            layer_name = f'encoder_1_{i}'
            self.layers[layer_name] = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_head, \
                                                                dim_feedforward=self.dim_ff, batch_first=True, dropout=self.dropout_ff)
        
        for i in range(self.n_enc2):
            layer_name = f'encoder_2_{i}'
            self.layers[layer_name] = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_head, \
                                                                dim_feedforward=self.dim_ff, batch_first=True, dropout=self.dropout_ff)

        lin_size = self.compute_output_shape()
        self.linear = nn.Linear(lin_size, self.n_classes)

    def compute_output_shape(self):

        assert self.concat_mode < 4, f"Concat mode can't be > 3, {self.concat_mode}"
        assert self.select_mode < 4, f"Select mode can't be > 3, {self.select_mode}"

        if self.concat_mode == 0 or (self.n_enc1 == 0 and self.n_enc2 != 0) or (self.n_enc2 == 0 and self.n_enc1 != 0):
            out_shape = [self.n_patch, self.d_model, self.n_dims]
        elif self.concat_mode == 1:
            out_shape = [self.n_patch * 2, self.d_model, self.n_dims]
        elif self.concat_mode == 2:
            out_shape = [self.n_patch, self.d_model * 2, self.n_dims]
        elif self.concat_mode == 3:
            out_shape = [self.n_patch, self.d_model, self.n_dims * 2]

        if self.select_mode == 0:
            pass
        else:
            out_shape[self.select_mode-1] = 1

        return out_shape[0] * out_shape[1] * out_shape[2]


    def concat_select(self, x1, x2):

        if self.n_enc1 == 0 and self.n_enc2 != 0:
            x = x2
        elif self.n_enc2 == 0 and self.n_enc1 != 0:
            x = x1
        else:
            if self.concat_mode == 0:
                x = x1 + x2
            else:
                x = torch.concat((x1, x2), axis=self.concat_mode) 

        if self.select_mode == 0:
            return x
        elif self.select_mode == 1:
            return x[:, 0, :, :]
        elif self.select_mode == 2:
            return x[:, :, 0, :]
        elif self.select_mode == 3:
            return x[:, :, :, 0]

    
    def forward(self, x):
        
        batch_size = x.shape[0]
        if self.embedding_mode:
            x = x.reshape(-1, self.d_model)
            x = self.emb(x)
        x = x.reshape(batch_size, self.d_model, self.n_patch, self.n_dims)

        x1 = x.permute(0, 2, 3, 1).reshape(-1, self.n_dims, self.d_model)
        x2 = (x.permute(0, 3, 2, 1) + self.pos_v).reshape(-1, self.n_patch, self.d_model)

        for layer_name, layer in self.layers.items():
            if layer_name[:-2] == 'encoder_1':
                x1 = layer(x1)
            else:
                x2 = layer(x2)

        x1 = x1.reshape(batch_size, self.n_patch, self.d_model, self.n_dims)
        x2 = x2.reshape(batch_size, self.n_dims, self.n_patch, self.d_model).permute(0, 2, 3, 1)

        x = self.concat_select(x1, x2)
        
        x = x.flatten(start_dim=1)
        x = self.linear(x)

        return x      