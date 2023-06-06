'''
This file contains the implementation of the TiDE model.
Original paper: https://arxiv.org/pdf/2304.08424.pdf
'''

import torch
import torch.nn as nn
from einops import rearrange

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

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
    
    
class ResidualBlock(nn.Module):
    '''
    The Residual block is the basic building block of TiDE.
    In feature projection, the out_features, is usually lower than in_features.
    '''
    def __init__(self, in_features, hid_features, out_features, dropout_prob=0.3 ,layer_norm=True):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, hid_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hid_features, out_features)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.norm = nn.LayerNorm(out_features)
        self.layer_norm = layer_norm
        # Shortcut connection
        self.shortcut = nn.Identity()
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out += self.shortcut(residual)
        if self.layer_norm:
            out = self.norm(out)
        return out
    

class TiDEEncoder(nn.Module):
    '''
    The encoder block of TiDE.
    Consists of a stack of residual blocks.

    Note:
    The encoder internal layer sizes are all set to hiddenSize and 
    the total number of layers in the encoder is set to ne(numEncoderLayers)

    The input to the encoder layer has to be stacked and flattened, 
    past and future projected covariates: X(1, L + H) has to be projected to lower dimension
    '''
    def __init__(self, in_features, hid_features, out_features, drop_prob, num_blocks=1,layer_norm=True):
        super(TiDEEncoder, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(ResidualBlock(in_features, hid_features, out_features, drop_prob,layer_norm))
            in_features = out_features

    def forward(self, x):
        '''
        Here x is the concatenation of the following three elements:
        1. past and future projected covariates: X(1, L + H)
        2. past of the time seires: Y(1, L)
        3. static covariates: a(i)
        '''
        for block in self.blocks:
            x = block(x)

        return x
    

class TiDEDenseDecoder(nn.Module):
    '''
    The dense decoder block of TiDE.
    It maps the encoded hidden representations into future predictions of time series
    Also consists of a stack of residual blocks.

    Note:
    The hidden layer size of the dense encoder is the same as the hidden layer size of the encoder.
    The decoder output size has to be a vector with size p * H, where H is future time steps and p is the expected dimension of decoder output of a single time step vector.
    '''
    def __init__(self, in_features, hid_features, out_features, drop_prob, num_blocks=1,layer_norm=True):
        super(TiDEDenseDecoder, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(ResidualBlock(in_features, hid_features, out_features, drop_prob,layer_norm))
            in_features = out_features

    def forward(self, x):
        '''
        Here x is the encoded hidden representation of the time series.
        '''
        for block in self.blocks:
            x = block(x)

        return x
    
class TiDETemporalDecoder(nn.Module):
    '''
    The temporal decoder block of TiDE.

    A residual block with output size 1 (the prediction), 
    that maps the decoded vector at time step t concatenated with the corresponding projected covariate.

    This can be useful if some covariated have a strong direct effect on a particular time-step's actual value
    '''
    def __init__(self, in_features, hid_features, out_features, drop_prob,layer_norm=True):
        super(TiDETemporalDecoder, self).__init__()
        self.residual = ResidualBlock(in_features, hid_features, out_features, drop_prob,layer_norm)

    def forward(self, x):
        '''
        Here x is the concatenation of the following two elements:
        1. decoded vector at time step t: y(t)
        2. corresponding projected covariate: X(t)
        '''
        return self.residual(x)
    
class TiDEModel(nn.Module):
    '''
    TiDE model, consisting of the encoder, dense decoder and temporal decoder.

    The input data consists of three parts:
    1. Lookback y(1: L)
    2. Static attributes a(i)
    3. Dynamic covariates X(1: L + H)
    '''
    def __init__(self, sizes, args):
        super(TiDEModel, self).__init__()
        self.lookback_shape = sizes['lookback']
        self.attr_shape = sizes['attr']
        self.dynCov_shape = sizes['dynCov']
        self.label_len = args.lookback_len  # lookback length L
        self.pred_len = args.pred_len       # prediction length H
        self.args = args

        # hyperparameters
        self.feat_size = args.feat_size
        self.hidden_size = args.hidden_size
        self.num_encoder_layers = args.num_encoder_layers
        self.num_decoder_layers = args.num_decoder_layers
        self.decoder_output_dim = args.decoder_output_dim
        self.temporal_decoder_hidden = args.temporal_decoder_hidden
        self.drop_prob = args.drop_prob
        self.layer_norm = args.layer_norm
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.revin=args.revin
        
        self.RevIN = RevIN(self.lookback_shape[1])
        
        # the concat shape is the combined size of three flattened tensor
        self.concat_shape = self.label_len * self.lookback_shape[1] + self.attr_shape[0] * self.attr_shape[1] + self.dynCov_shape[0] * self.feat_size
        # modules
        self.featproj = ResidualBlock(self.dynCov_shape[1], self.hidden_size, self.feat_size, self.drop_prob,self.layer_norm)
        self.encoder = TiDEEncoder(self.concat_shape, self.hidden_size, self.hidden_size, self.drop_prob, self.num_encoder_layers,self.layer_norm)
        self.denseDecoder = TiDEDenseDecoder(self.hidden_size, self.hidden_size, self.decoder_output_dim * self.pred_len, self.drop_prob, self.num_decoder_layers,self.layer_norm)
        self.temporalDecoder = TiDETemporalDecoder(self.feat_size + self.decoder_output_dim, self.temporal_decoder_hidden,self.lookback_shape[1], self.drop_prob,self.layer_norm)
        
        self.linear = nn.Linear(self.label_len, self.pred_len)

    def forward(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        '''
        batch_x: attributes (b,L+H,N)
        batch_y: look back (B,L,N) + ans (B,L:L+H,N)
        batch_x_mark: Dynamic covariates

        Here lookback, attr and dynCov are the three input data parts.
        Each input has three dimensions: (batch_size, time_steps, feature_size).
        '''
        # Feature projection
        proj_feature = self.featproj(batch_y_mark)
        batch_y_lookback = batch_y[:, : self.label_len, :]#Lookback (B,L,N)
    
        if self.revin:
            batch_y_lookback = self.RevIN(batch_y_lookback,mode='norm')
        # Encoder processing
        # first flatten and concatenate the input data
        proj_feature_ = proj_feature.view(self.batch_size, -1)
        batch_x_ = batch_x.view(self.batch_size, -1)
        batch_y0 = batch_y_lookback.view(self.batch_size, -1)
        
        encoder_input = torch.cat((proj_feature_, batch_x_, batch_y0), dim=1)

        encoded = self.encoder(encoder_input.float())

        # Dense decoder processing
        denseDecoded = self.denseDecoder(encoded)

        # unflatten the decoded vector
        denseDecoded = denseDecoded.view(self.batch_size, self.pred_len, self.decoder_output_dim)
        pred_feat = proj_feature[:, -self.pred_len:, :]
        # stack the two
        denseDecoded = torch.cat((denseDecoded, pred_feat), dim=2)
        
        # Temporal decoder
        temporalDecoded = self.temporalDecoder(denseDecoded)
        
        batch_y_lookback = rearrange(batch_y_lookback, 'b l n -> (b n) l')  # [B,L,N] -> [B*N,L] Guarantee channel independence
        # lookback residual
        res_lookback = self.linear(batch_y_lookback)#->(B*N,H)
        res_lookback = rearrange(res_lookback, '(b n) h -> b h n', b=self.args.batch_size)# [B*N,H] -> [B,H,N] 
        #print(res_lookback.shape)
        pred = temporalDecoded + res_lookback
        if self.revin:
            pred = self.RevIN(pred,mode='denorm')
        ans = batch_y[:, -self.pred_len:, :]
        #ans:(B,H,N)
        return pred, ans
