import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from src.models.adapter_nets import init_ssf_scale_shift, ssf_ada
from src.models.model_utils import trunc_normal_

class NormAndLinear(nn.Module):
    def __init__(self, dim, num_classes, adapter=None, dropout=0., **kwargs):
        super().__init__()

        self.adapter = adapter

        if adapter and adapter.name == 'ssf':
            self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(dim)
        self.layer_norm = nn.LayerNorm(dim)
        
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, num_classes)
        )
    def forward(self, x):
        x = self.layer_norm(x)
        if self.adapter and self.adapter.name == 'ssf':
            x = ssf_ada(x, self.ssf_scale_1, self.ssf_shift_1)
        return self.mlp_head(x)

class Linear(nn.Module):
    def __init__(self, dim, num_classes, **kwargs):
        super().__init__()
        self.linear = nn.Linear(dim, num_classes)
        
    def forward(self, x):
        return self.linear(x)
                

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0., attn_dim=0.):
        super().__init__()
        self.heads = heads
        if attn_dim == 0:
            attn_dim = dim

        self.scale = attn_dim ** -0.5

        self.to_qkv = nn.Linear(dim, attn_dim * 3, bias = True)
        self.to_out = nn.Sequential(
            nn.Linear(attn_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, recurrent_steps, heads, dropout, depth=1, attn_dim=0., mlp_ratio=4):
        super().__init__()
        self.recurrent_steps = recurrent_steps
        self.depth = depth
        self.layers = nn.ModuleList([])
        for _ in range(self.depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dropout = dropout, attn_dim=attn_dim))),
                Residual(PreNorm(dim, FeedForward(dim, int(dim * mlp_ratio), dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for j in range(self.depth):
            for i in range(self.recurrent_steps):
                x = self.layers[j][0](x, mask = mask) # att
                x = self.layers[j][1](x) # ffn
        return x

class Ree(nn.Module):
    def __init__(self, 
        recurrent_steps, 
        heads, 
        depth, 
        base_model, 
        num_classes, 
        adapter=None, 
        dropout = 0., 
        emb_dropout = 0., 
        modulation=True, 
        exit_head='normlinear', # TODO: Pass in cls instead
        attn_dim=16, mlp_ratio=2,
        **kwargs):
        super().__init__()
        self.recurrent_steps = recurrent_steps 
        self.heads = heads 
        self.depth =  depth
        self.base_model = base_model
        self.num_classes = num_classes
        self.modulation = modulation # cls token modulation

        if 'base' in self.base_model:
            self.dim = 768
            self.pos_embedding = nn.Parameter(torch.zeros(1, 12+1, self.dim))
        elif 'small' in self.base_model:
            self.dim = 384
            self.pos_embedding = nn.Parameter(torch.zeros(1, 12+1, self.dim))
        elif 'XXS24' in self.base_model or 'vim' in self.base_model:
            self.dim = 192
            self.pos_embedding = nn.Parameter(torch.zeros(1, 24+1, self.dim))
        elif 'tiny' in self.base_model:
            self.dim = 192
            self.pos_embedding = nn.Parameter(torch.zeros(1, 12+1, self.dim))
        else: # resnet
            self.dim = 512
            self.pos_embedding = nn.Parameter(torch.zeros(1, 4+1, self.dim))
        
        self.client_token = nn.Parameter(torch.zeros(1, 1, self.dim))
        self.dropout = nn.Dropout(emb_dropout)

        trunc_normal_(self.pos_embedding, std=.02)
        trunc_normal_(self.client_token, std=.02)

        self.transformer = Transformer(self.dim, self.recurrent_steps, self.heads, dropout, depth=self.depth, attn_dim=attn_dim, mlp_ratio=mlp_ratio)

        exit_funcs = {'normlinear': NormAndLinear, 'linear': Linear}
        exit_func = exit_funcs[exit_head]

        self.head = exit_func(self.dim, self.num_classes, adapter=adapter)

    def forward(self, features, **kwargs):
        # features are cls_tokens
        b, n, _ = features.shape
        last_cls_token = features[:, -1]
        
        client_token = self.client_token.expand(b, -1, -1)
        x = torch.cat((client_token, features), dim=1)

        x += self.pos_embedding[:, :(n + 1)]

        x = self.dropout(x)

        m = self.transformer(x)

        return m

