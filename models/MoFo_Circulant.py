import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import math


class MoFo_Circulant_Backbone(nn.Module):
    def __init__(self, dim, cycle, head, lambda_init=0.1, use_causal_mask=False):
        super(MoFo_Circulant_Backbone, self).__init__()
        self.dim = dim
        self.attn = MoFo_Circulant_Attention(dim, cycle, head, lambda_init, use_causal_mask)
        self.ffn = SwiGLU_FFN(dim, dim)

        self.attn_norm = RMSNorm(dim, bias=True)
        self.ffn_norm = RMSNorm(dim, bias=True)

    def forward(self, x):
        x = self.attn(self.attn_norm(x)) + x
        x = self.ffn(self.ffn_norm(x)) + x
        return x


class MoFo_Circulant_Attention(nn.Module):
    def __init__(self, dim, cycle=24, head=4, lambda_init=0.1, use_causal_mask=False):
        super(MoFo_Circulant_Attention, self).__init__()
        self.head_num = head
        self.head_dim = dim // head
        assert dim % head == 0, "dim must be divisible by head"
        self.transformation = nn.Sequential(
            nn.Linear(dim, 3 * dim),
            nn.Unflatten(dim=-1, unflattened_size=(self.head_num, 3 * self.head_dim))
        )
        self.outer = nn.Sequential(
            nn.Flatten(start_dim=-2, end_dim=-1),
            nn.Linear(dim, dim)
        )

        self.cycle = cycle
        self.use_causal_mask = use_causal_mask
        self.norm = self.head_dim ** (-0.5)

        self.log_lambda = nn.Parameter(torch.tensor(math.log(lambda_init + 1e-8)), requires_grad=True)

        self.token_reweight_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
        )

    @property
    def lambda_reg(self):
        return F.softplus(self.log_lambda)

    def compute_circulant_mean(self, attn_matrix):
        B, H, N, _ = attn_matrix.shape
        device = attn_matrix.device
        dtype = attn_matrix.dtype

        if self.use_causal_mask:
            mask = torch.tril(torch.ones(N, N, device=device, dtype=dtype)).unsqueeze(0).unsqueeze(0)
        else:
            mask = torch.ones(N, N, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)

        diag_sums = torch.zeros(B, H, 2 * N - 1, device=device, dtype=dtype)
        diag_counts = torch.zeros(2 * N - 1, device=device, dtype=dtype)

        idx_i = torch.arange(N, device=device)
        idx_j = torch.arange(N, device=device)
        offset = idx_j.unsqueeze(0) - idx_i.unsqueeze(1)
        offset_idx = offset + (N - 1)

        masked_attn = attn_matrix * mask
        for k in range(2 * N - 1):
            k_offset = k - (N - 1)
            if k_offset >= 0:
                row_start = 0
                row_end = N - k_offset
                col_start = k_offset
                col_end = N
            else:
                row_start = -k_offset
                row_end = N
                col_start = 0
                col_end = N + k_offset
            diag_sums[:, :, k] = masked_attn[:, :, row_start:row_end, col_start:col_end].sum(dim=-1).sum(dim=-1)
            diag_counts[k] = mask[0, 0, row_start:row_end, col_start:col_end].sum()

        diag_counts = diag_counts.clamp(min=1)
        diag_means = diag_sums / diag_counts.unsqueeze(0).unsqueeze(0)

        circulant = torch.zeros_like(attn_matrix)
        for k in range(2 * N - 1):
            k_offset = k - (N - 1)
            if k_offset >= 0:
                row_start = 0
                row_end = N - k_offset
                col_start = k_offset
                col_end = N
            else:
                row_start = -k_offset
                row_end = N
                col_start = 0
                col_end = N + k_offset
            circulant[:, :, row_start:row_end, col_start:col_end] = diag_means[:, :, k].unsqueeze(-1).unsqueeze(-1)

        return circulant

    def circulant_regularization_loss(self, attn_matrix):
        circulant_approx = self.compute_circulant_mean(attn_matrix.detach())
        return F.mse_loss(attn_matrix, circulant_approx)

    def forward(self, x):
        B_C, T, D = x.shape

        token_reweight = self.token_reweight_proj(x)

        query, key, value = torch.chunk(self.transformation(x), 3, dim=-1)

        attn_scores = query.transpose(1, 2) @ key.permute(0, 2, 3, 1) * self.norm

        if self.use_causal_mask:
            causal_mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
            attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=-1)

        attn_out = attn_weights @ value.transpose(1, 2)
        attn_out = self.outer(attn_out.transpose(1, 2))

        attn_out = attn_out * token_reweight

        return attn_out


class SwiGLU_FFN(nn.Module):
    def __init__(self, dim_in, dim_out, expand_ratio=4, dropout=0.3, norm=None):
        super(SwiGLU_FFN, self).__init__()
        self.W1 = nn.Linear(dim_in, expand_ratio * dim_in)
        self.W2 = nn.Linear(dim_in, expand_ratio * dim_in)
        self.W3 = nn.Linear(expand_ratio * dim_in, dim_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.W3(self.dropout(F.silu(self.W1(x)) * self.W2(x)))


class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias
        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)
        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)
            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size
        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)
        if self.bias:
            return self.scale * x_normed + self.offset
        return self.scale * x_normed


class MoFo_Circulant(nn.Module):
    def __init__(self, configs, individual=False):
        super(MoFo_Circulant, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len

        self.channels = configs.enc_in
        self.individual = individual
        self.dim = configs.d_model
        self.norm = RevIN(self.channels, eps=1e-5, affine=True)
        self.periodic = configs.periodic
        self.head = configs.head
        self.periodic_index = nn.Parameter(torch.arange(0, self.periodic), requires_grad=False)
        self.periodic_num = math.ceil(self.seq_len / self.periodic)
        try:
            self.layers = configs.d_layers
        except:
            self.layers = 1
        self.if_bias = configs.bias
        self.if_cias = configs.cias
        self.padding_num = self.seq_len % self.periodic

        lambda_init = getattr(configs, 'lambda_init', 0.1)
        use_causal_mask = getattr(configs, 'use_causal_mask', False)

        self.input = nn.Sequential(
            nn.Unflatten(dim=-1, unflattened_size=(self.periodic, self.periodic_num)),
            nn.Linear(self.periodic_num, self.dim),
        )
        self.input_multiperiod = nn.Sequential(
            nn.Unflatten(dim=-1, unflattened_size=(self.periodic, self.periodic_num)),
            nn.Linear(self.periodic_num, self.dim),
        )
        if self.if_bias:
            self.bias = nn.Parameter(torch.empty(1, self.channels, 1, self.dim))
            nn.init.xavier_normal_(self.bias)
        if self.if_cias:
            self.cias = nn.Parameter(torch.empty(self.periodic, self.dim))
            nn.init.xavier_normal_(self.cias)
            self.ciasW = nn.Parameter(torch.empty(7, self.dim))
            nn.init.xavier_normal_(self.ciasW)
        self.output = nn.Sequential(
            nn.Flatten(start_dim=-2, end_dim=-1),
            nn.Linear(self.dim * self.periodic, self.pred_len),
        )
        self.regression = nn.Linear(self.periodic, self.pred_len)

        self.MoFo_Backbone = nn.Sequential(*[
            MoFo_Circulant_Backbone(self.dim, self.periodic, self.head, lambda_init, use_causal_mask)
            for _ in range(self.layers)
        ])

    def encoder(self, x, periodic_position, periodic_positionW):
        x = self.norm(x, mode='norm').permute(0, 2, 1)
        if self.padding_num:
            x = torch.concat([x[..., self.padding_num:self.periodic], x], dim=-1)

        x = self.input(x) + self._ias(self.periodic, periodic_position)

        x = self.MoFo_Backbone(x.reshape(-1, self.periodic, self.dim))
        x = self.output(x)
        x = self.norm(x.reshape(-1, self.channels, self.pred_len).permute(0, 2, 1),
                       mode='denorm')
        return x

    def _ias(self, p, periodic_position, periodic_positionW=None):
        out = 0
        if self.if_cias:
            c_index = (periodic_position - self.periodic_index.unsqueeze(0)) % p
            cias = self.cias[c_index.long()].unsqueeze(1)
            out = out + cias
        if self.if_bias:
            out = out + self.bias
        return out

    def get_circulant_reg_loss(self):
        total_loss = 0.0
        for module in self.MoFo_Backbone:
            if isinstance(module, MoFo_Circulant_Backbone):
                total_loss += module.attn.lambda_reg
        return total_loss

    def forecast(self, x_enc, periodic_position, periodic_positionW):
        return self.encoder(x_enc, periodic_position, periodic_positionW)

    def imputation(self, x_enc):
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        return self.encoder(x_enc)

    def classification(self, x_enc):
        enc_out = self.encoder(x_enc)
        output = enc_out.reshape(enc_out.shape[0], -1)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.periodic == 24:
            periodic_position = torch.round((x_mark_enc[:, -1, 0:1] + 0.5) * (24 - 1))
        elif self.periodic == 96:
            periodic_position = torch.round((x_mark_enc[:, -1, 2:3] + 0.5) * (24 - 1)) * 4 + \
                                torch.round((x_mark_enc[:, -1, 1:2] + 0.5) * (60 - 1)) / 15
        elif self.periodic == 144:
            periodic_position = torch.round((x_mark_enc[:, -1, 2:3] + 0.5) * (24 - 1)) * 6 + \
                                torch.round((x_mark_enc[:, -1, 1:2] + 0.5) * (60 - 1)) / 10
        elif self.periodic == 288:
            periodic_position = torch.round((x_mark_enc[:, -1, 2:3] + 0.5) * (24 - 1)) * 12 + \
                                torch.round((x_mark_enc[:, -1, 1:2] + 0.5) * (60 - 1)) / 5
        else:
            periodic_position = None

        if x_mark_enc.shape[-1] == 4:
            periodic_positionW = x_mark_enc[..., 1]
        elif x_mark_enc.shape[-1] == 6:
            periodic_positionW = x_mark_enc[..., 3]
        else:
            periodic_positionW = None

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, periodic_position, periodic_positionW)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc)
            return dec_out
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out
        return None


class Linear(nn.Module):
    def __init__(self, seq_len, pred_len, bias=False):
        super(Linear, self).__init__()
        self.weight = nn.Parameter((1 / seq_len) * torch.ones(1, pred_len, seq_len))
        if bias:
            self.bias = nn.Parameter(torch.zeros(1, pred_len, 1))
        else:
            self.bias = 0

    def forward(self, x, relative_cp=1):
        return (relative_cp * self.weight) @ x + self.bias


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x
