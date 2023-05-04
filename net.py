import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# B: Batchsize
# L: Lookback
# H: Horizon
# N: the number of series
# r: the number of covariates for each series
# r_hat: temporalWidth in the paper, i.e., \hat{r} << r
# p: decoderOutputDim in the paper
# hidden_dim: hiddenSize in the paper
class ResidualBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        self.linear_1 = nn.Linear(in_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, out_dim)
        self.linear_res = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.layernorm = nn.LayerNorm(out_dim)

    def forward(self, x):
        # x: [B,L,in_dim] or [B,in_dim]
        h = F.relu(self.linear_1(x))  # [B,L,in_dim] -> [B,L,hidden_dim] or [B,in_dim] -> [B,hidden_dim]
        h = self.dropout(self.linear_2(h))  # [B,L,hidden_dim] -> [B,L,out_dim] or [B,hidden_dim] -> [B,out_dim]
        res = self.linear_res(x)  # [B,L,in_dim] -> [B,L,out_dim] or [B,in_dim] -> [B,out_dim]
        out = self.layernorm(h+res)  # [B,L,out_dim] or [B,out_dim] 

        # out: [B,L,out_dim] or [B,out_dim]
        return out
    
class Encoder(nn.Module):
    def __init__(self, layer_num, hidden_dim, r, r_hat, L, H, featureProjectionHidden):
        super(Encoder, self).__init__()
        self.encoder_layer_num = layer_num
        self.horizon = H
        self.feature_projection = ResidualBlock(r, featureProjectionHidden, r_hat)
        self.first_encoder_layer = ResidualBlock(L + 1 + (L + H) * r_hat, hidden_dim, hidden_dim)
        self.other_encoder_layers = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim, hidden_dim) for _ in range(layer_num-1)
            ])

    def forward(self, x, covariates, attributes):
        # x: [B*N,L], covariates: [B*N,1], attributes: [B*N,L+H,r]

        # Feature Projection
        covariates = self.feature_projection(covariates)  # [B*N,L+H,r] -> [B*N,L+H,r_hat]
        covariates_future = covariates[:, -self.horizon:, :]  # [B*N,H,r_hat]

        # Flatten
        covariates_flat = rearrange(covariates, 'b l r -> b (l r)')  # [B*N,L+H,r_hat] -> [B*N,(L+H)*r_hat]

        # Concat
        e = torch.cat([x, attributes, covariates_flat], dim=1)  # [B*N,L+1+(L+H)*r_hat]

        # Dense Encoder
        e = self.first_encoder_layer(e)  # [B*N,L+1+(L+H)*r_hat] -> [B*N,hidden_dim]
        for i in range(self.encoder_layer_num-1):
            e = self.other_encoder_layers[i](e)  # [B*N,hidden_dim] -> [B*N,hidden_dim]

        # e: [B*N,hidden_dim], covariates_future: [B*N,H,r_hat]
        return e, covariates_future

    
class Decoder(nn.Module):
    def __init__(self, layer_num, hidden_dim, r_hat, H, p, temporalDecoderHidden):
        super(Decoder, self).__init__()
        self.decoder_layer_num = layer_num
        self.horizon = H
        self.last_decoder_layer = ResidualBlock(hidden_dim, hidden_dim, p * H)
        self.other_decoder_layers = nn.ModuleList([
                ResidualBlock(hidden_dim, hidden_dim, hidden_dim) for _ in range(layer_num-1)
            ])
        self.temporaldecoder = ResidualBlock(p + r_hat, temporalDecoderHidden, 1)

    def forward(self, e, covariates_future):
        # e: [B*N,hidden_dim], covariates_future: [B*N,H,r_hat]

        # Dense Decoder
        for i in range(self.decoder_layer_num-1):
            e = self.other_decoder_layers[i](e)  # [B*N,hidden_dim] -> [B*N,hidden_dim]
        g = self.last_decoder_layer(e)  # [B*N,hidden_dim] -> [B*N,p*H]

        # Unflatten
        matrixD = rearrange(g, 'b (h p) -> b h p', h=self.horizon)  # [B*N,p*H] -> [B*N,H,p]

        # Stack
        out = torch.cat([matrixD, covariates_future], dim=-1)  # [B*N,H,p+r_hat]

        # Temporal Decoder
        out = self.temporaldecoder(out)  # [B*N,H,p+r_hat] -> [B*N,H,1]
        
        # out: [B*N,H,1]
        return out
    
class TiDE(nn.Module):
    def __init__(
            self,
            L,
            H,
            r,
            r_hat,
            p,
            hidden_dim,
            encoder_layer_num,
            decoder_layer_num,
            featureProjectionHidden,
            temporalDecoderHidden,
        ):
        super(TiDE, self).__init__()
        self.encoder = Encoder(encoder_layer_num, hidden_dim, r, r_hat, L, H, featureProjectionHidden)
        self.decoder = Decoder(decoder_layer_num, hidden_dim, r_hat, H, p, temporalDecoderHidden)
        self.residual = nn.Linear(L, H)

    def forward(self, x, covariates, attributes):
        # x: [B,L,N], covariates: [B,L+H,N,r], attributes: [B,N,1]
        batch_size = x.size(0)
        
        # Channel Independence: Convert Multivariate series to Univariate series
        x = rearrange(x, 'b l n -> (b n) l')  # [B,L,N] -> [B*N,L]
        covariates = rearrange(covariates, 'b l n r -> (b n) l r')  # [B,L+H,N,r] -> [B*N,L+H,r]
        attributes = rearrange(attributes, 'b n 1 -> (b n) 1')  # [B,N,1] -> [B*N,1]
        
        # Encoder
        e, covariates_future = self.encoder(x, covariates, attributes)

        # Decoder
        out = self.decoder(e, covariates_future)  # out: [B*N,H,1]

        # Global Residual
        prediction = out.squeeze(-1) + self.residual(x)  # prediction: [B*N,H]

        # Reshape
        prediction = rearrange(prediction, '(b n) h -> b h n', b=batch_size)  # [B*N,H] -> [B,H,N]

        # prediction: [B,H,N]
        return prediction