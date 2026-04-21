import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import math


class MoFo_Circulant_DP_Backbone(nn.Module):
    def __init__(self, dim, cycle, head, lambda_init=0.1, use_causal_mask=False,
                 use_dual_path=False, decomp_mode='stl', dual_path_period=0,
                 trend_mode='linear', dual_path_alpha_init=0.5):
        super(MoFo_Circulant_DP_Backbone, self).__init__()
        self.dim = dim
        self.use_dual_path = use_dual_path

        self.attn = MoFo_Circulant_DP_Attention(dim, cycle, head, lambda_init, use_causal_mask)
        self.ffn = SwiGLU_FFN_DP(dim, dim)

        self.attn_norm = RMSNorm_DP(dim, bias=True)
        self.ffn_norm = RMSNorm_DP(dim, bias=True)

        if use_dual_path:
            self.dual_path = DualPathSeriesDecomposer(
                dim=dim,
                decomp_mode=decomp_mode,
                period=dual_path_period,
                trend_mode=trend_mode,
                alpha_init=dual_path_alpha_init,
            )

    def forward(self, x, raw_x=None):
        if self.use_dual_path:
            x = self.dual_path(x, raw_x=raw_x)
        x = self.attn(self.attn_norm(x)) + x
        x = self.ffn(self.ffn_norm(x)) + x
        return x


class MoFo_Circulant_DP_Attention(nn.Module):
    def __init__(self, dim, cycle=24, head=4, lambda_init=0.1, use_causal_mask=False):
        super(MoFo_Circulant_DP_Attention, self).__init__()
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


class SeriesDecomp(nn.Module):
    def __init__(self, decomp_mode='stl', period=0, min_period=2):
        super(SeriesDecomp, self).__init__()
        self.decomp_mode = decomp_mode
        self.period = period
        self.min_period = min_period
        self._detected_period = None

    def detect_period_fft_acf(self, x):
        B, N, D = x.shape
        device = x.device
        dtype = x.dtype

        x_mean = x.mean(dim=1, keepdim=True)
        x_centered = x - x_mean

        pad_len = N * 4
        x_padded = F.pad(x_centered, (0, 0, 0, pad_len - N), mode='constant', value=0)

        X_fft = torch.fft.rfft(x_padded, dim=1)
        psd = X_fft.abs().pow(2).mean(dim=-1)

        psd_mean = psd.mean(dim=0)
        psd_np = psd_mean.detach().cpu().numpy()

        top_k = 5
        candidates = []
        for i in range(2, len(psd_np) - 1):
            if psd_np[i] > psd_np[i - 1] and psd_np[i] > psd_np[i + 1]:
                candidates.append((i, psd_np[i]))
        candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = candidates[:top_k]

        period_candidates = []
        for freq_bin, _ in candidates:
            if freq_bin > 0:
                p = int(round(pad_len / freq_bin))
                if self.min_period <= p <= N // 2:
                    period_candidates.append(p)

        if not period_candidates:
            return max(self.min_period, N // 4)

        x_det = x_centered.detach()
        best_period = period_candidates[0]
        best_acf = -float('inf')

        for p in period_candidates:
            for delta in [-1, 0, 1]:
                pd = p + delta
                if pd < self.min_period or pd >= N:
                    continue
                shift = pd
                if shift >= N:
                    continue
                a = x_det[:, :N - shift, :]
                b = x_det[:, shift:, :]
                acf_score = (a * b).sum() / (a.pow(2).sum().clamp(min=1e-8)).sqrt().clamp(min=1e-8)
                acf_val = acf_score.mean().item()
                if acf_val > best_acf:
                    best_acf = acf_val
                    best_period = pd

        return best_period

    def forward_stl(self, x, raw_x=None):
        B, N, D = x.shape
        device = x.device
        dtype = x.dtype

        if self.period > 0:
            period = self.period
        elif raw_x is not None:
            period = self.detect_period_fft_acf(raw_x)
            self._detected_period = period
        else:
            period = self.detect_period_fft_acf(x)
            self._detected_period = period

        trend_ema = self._ema(x, alpha=0.1)
        residual = x - trend_ema

        X_fft = torch.fft.rfft(x, dim=1)
        freqs = torch.fft.rfftfreq(N, device=device)

        period_freq = 1.0 / period
        sigma = 1.0 / (period * 2.0)
        freq_dist = (freqs - period_freq).abs()
        seasonal_mask = torch.exp(-0.5 * (freq_dist / sigma).pow(2))
        seasonal_mask = seasonal_mask.unsqueeze(0).unsqueeze(-1)

        harmonic_mask = torch.zeros_like(seasonal_mask)
        for h in range(1, 4):
            hf = period_freq * h
            if hf < 0.5:
                hd = (freqs - hf).abs()
                harmonic_mask = harmonic_mask + torch.exp(-0.5 * (hd / sigma).pow(2)).unsqueeze(0).unsqueeze(-1)

        total_seasonal_mask = (seasonal_mask + harmonic_mask).clamp(max=1.0)
        trend_mask = 1.0 - total_seasonal_mask

        seasonal_fft = X_fft * total_seasonal_mask
        trend_fft = X_fft * trend_mask

        seasonal = torch.fft.irfft(seasonal_fft, n=N, dim=1)
        trend = torch.fft.irfft(trend_fft, n=N, dim=1)

        seasonal = self._circular_shift_blend(seasonal)

        return trend, seasonal

    def forward_ma(self, x, raw_x=None):
        B, N, D = x.shape
        kernel_size = max(3, N // 10)
        if kernel_size % 2 == 0:
            kernel_size += 1

        x_padded = F.pad(x.permute(0, 2, 1), (kernel_size // 2, kernel_size // 2), mode='replicate')
        trend = F.avg_pool1d(x_padded, kernel_size=kernel_size, stride=1)
        trend = trend.permute(0, 2, 1)
        seasonal = x - trend
        return trend, seasonal

    def _ema(self, x, alpha=0.1):
        B, N, D = x.shape
        ema = torch.zeros_like(x)
        ema[:, 0, :] = x[:, 0, :]
        for t in range(1, N):
            ema[:, t, :] = alpha * x[:, t, :] + (1 - alpha) * ema[:, t - 1, :]
        return ema

    def _circular_shift_blend(self, x, blend_frac=0.1):
        B, N, D = x.shape
        blend_len = max(1, int(N * blend_frac))
        if blend_len >= N:
            return x
        ramp = torch.linspace(0, 1, blend_len, device=x.device, dtype=x.dtype)
        ramp = ramp.unsqueeze(0).unsqueeze(-1)
        x[:, :blend_len, :] = x[:, :blend_len, :] * ramp + x[:, -blend_len:, :].flip(1) * (1 - ramp)
        return x

    def forward(self, x, raw_x=None):
        if self.decomp_mode == 'stl':
            return self.forward_stl(x, raw_x)
        else:
            return self.forward_ma(x, raw_x)


class TrendProjector(nn.Module):
    def __init__(self, d_model, mode='linear'):
        super(TrendProjector, self).__init__()
        if mode == 'linear':
            self.proj = nn.Linear(d_model, d_model)
        elif mode == 'mlp':
            self.proj = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
            )

    def forward(self, x):
        return self.proj(x)


class Recomposer(nn.Module):
    def __init__(self, alpha_init=0.5):
        super(Recomposer, self).__init__()
        self._logit_alpha = nn.Parameter(torch.tensor(math.log(alpha_init / (1.0 - alpha_init + 1e-8)), dtype=torch.float32))

    @property
    def alpha(self):
        return torch.sigmoid(self._logit_alpha)

    def forward(self, trend_proj, seasonal_out):
        a = self.alpha
        return a * trend_proj + (1 - a) * seasonal_out


class DualPathSeriesDecomposer(nn.Module):
    def __init__(self, dim, decomp_mode='stl', period=0, trend_mode='linear', alpha_init=0.5):
        super(DualPathSeriesDecomposer, self).__init__()
        self.decomp = SeriesDecomp(decomp_mode=decomp_mode, period=period)
        self.trend_proj = TrendProjector(dim, mode=trend_mode)
        self.recomposer = Recomposer(alpha_init=alpha_init)
        self.norm = RMSNorm_DP(dim, bias=True)

    def forward(self, x, raw_x=None):
        trend_raw, seasonal_raw = self.decomp(x, raw_x=raw_x)
        trend_proj = self.trend_proj(trend_raw)
        seasonal_out = self.norm(seasonal_raw)
        fused = self.recomposer(trend_proj, seasonal_out)
        return fused


class SwiGLU_FFN_DP(nn.Module):
    def __init__(self, dim_in, dim_out, expand_ratio=4, dropout=0.3, norm=None):
        super(SwiGLU_FFN_DP, self).__init__()
        self.W1 = nn.Linear(dim_in, expand_ratio * dim_in)
        self.W2 = nn.Linear(dim_in, expand_ratio * dim_in)
        self.W3 = nn.Linear(expand_ratio * dim_in, dim_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.W3(self.dropout(F.silu(self.W1(x)) * self.W2(x)))


class RMSNorm_DP(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        super(RMSNorm_DP, self).__init__()
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


class MoFo_Circulant_DualPath(nn.Module):
    def __init__(self, configs, individual=False):
        super(MoFo_Circulant_DualPath, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len

        self.channels = configs.enc_in
        self.individual = individual
        self.dim = configs.d_model
        self.norm = RevIN_DP(self.channels, eps=1e-5, affine=True)
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
        use_dual_path = getattr(configs, 'use_dual_path', True)
        decomp_mode = getattr(configs, 'decomp_mode', 'stl')
        dual_path_period = getattr(configs, 'dual_path_period', 0)
        trend_mode = getattr(configs, 'trend_mode', 'mlp')
        dual_path_alpha_init = getattr(configs, 'dual_path_alpha_init', 0.5)

        self.use_dual_path = use_dual_path

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
            MoFo_Circulant_DP_Backbone(
                self.dim, self.periodic, self.head, lambda_init, use_causal_mask,
                use_dual_path=use_dual_path,
                decomp_mode=decomp_mode,
                dual_path_period=dual_path_period,
                trend_mode=trend_mode,
                dual_path_alpha_init=dual_path_alpha_init,
            )
            for _ in range(self.layers)
        ])

    def encoder(self, x, periodic_position, periodic_positionW, raw_x=None):
        x = self.norm(x, mode='norm').permute(0, 2, 1)
        if self.padding_num:
            x = torch.concat([x[..., self.padding_num:self.periodic], x], dim=-1)

        x = self.input(x) + self._ias(self.periodic, periodic_position)

        x_reshaped = x.reshape(-1, self.periodic, self.dim)

        if self.use_dual_path and raw_x is not None:
            x_normed = self.norm(x.reshape(-1, self.channels, self.periodic, self.dim).permute(0, 2, 1).reshape(-1, self.periodic, self.dim * self.channels), mode='norm') if False else raw_x
            for module in self.MoFo_Backbone:
                x_reshaped = module(x_reshaped, raw_x=raw_x)
        else:
            x_reshaped = self.MoFo_Backbone(x_reshaped)

        x = self.output(x_reshaped)
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
            if isinstance(module, MoFo_Circulant_DP_Backbone):
                total_loss += module.attn.lambda_reg
        return total_loss

    def forecast(self, x_enc, periodic_position, periodic_positionW, raw_x=None):
        return self.encoder(x_enc, periodic_position, periodic_positionW, raw_x=raw_x)

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

        raw_x = x_enc

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, periodic_position, periodic_positionW, raw_x=raw_x)
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


class Linear_DP(nn.Module):
    def __init__(self, seq_len, pred_len, bias=False):
        super(Linear_DP, self).__init__()
        self.weight = nn.Parameter((1 / seq_len) * torch.ones(1, pred_len, seq_len))
        if bias:
            self.bias = nn.Parameter(torch.zeros(1, pred_len, 1))
        else:
            self.bias = 0

    def forward(self, x, relative_cp=1):
        return (relative_cp * self.weight) @ x + self.bias


class RevIN_DP(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN_DP, self).__init__()
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
