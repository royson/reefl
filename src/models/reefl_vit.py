import os
import torch
import torch.nn as nn
import math
from functools import partial
from src.utils import get_func_from_config
from src.models.model_utils import trunc_normal_, prune
from src.models.adapter_nets import Adapter_Layer, LoRALinear, init_ssf_scale_shift, ssf_ada
from src.models.accumulator_nets import Ree

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., adapter=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.ffn_adapter_name = None
        if adapter:
            self.ffn_adapter_name = adapter.name
            if adapter.name in ['sa', 'pa']:
                self.ffn_adapter = Adapter_Layer(out_features, **adapter.args)
            elif adapter.name == 'ssf':
                self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(hidden_features)
                self.ssf_scale_2, self.ssf_shift_2 = init_ssf_scale_shift(out_features)
        
        
    def forward(self, x):
        if self.ffn_adapter_name == 'pa':
            res = self.ffn_adapter(x)

        x = self.fc1(x)
        if self.ffn_adapter_name == 'ssf':
            x = ssf_ada(x, self.ssf_scale_1, self.ssf_shift_1)

        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        if self.ffn_adapter_name == 'ssf':
            x = ssf_ada(x, self.ssf_scale_2, self.ssf_shift_2)

        x = self.drop(x)

        if self.ffn_adapter_name == 'pa':
            x = x + res

        if self.ffn_adapter_name == 'sa':
            x = self.ffn_adapter(x)

        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., adapter=None):
        super().__init__()
        attn_dim = dim
        if dim % num_heads != 0:
            attn_dim = math.ceil(dim / num_heads) * num_heads # round to nearest dim that is divisible by num_heads
        assert attn_dim % num_heads == 0
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        head_dim = attn_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        if adapter and adapter.name == 'lora':
            # apply to q and v only
            self.q = LoRALinear(dim, attn_dim, bias=qkv_bias, **adapter.args)
            self.k = nn.Linear(dim, attn_dim, bias=qkv_bias)
            self.v = LoRALinear(dim, attn_dim, bias=qkv_bias, **adapter.args)
        else:
            # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.q = nn.Linear(dim, attn_dim, bias=qkv_bias)
            self.k = nn.Linear(dim, attn_dim, bias=qkv_bias)
            self.v = nn.Linear(dim, attn_dim, bias=qkv_bias)

        self.attn_adapter_name = None
        if adapter:
            self.attn_adapter_name = adapter.name
            if adapter.name in ['sa', 'pa']:
                self.attn_adapter = Adapter_Layer(dim, **adapter.args)
            elif adapter.name == 'ssf':
                self.ssf_scale_q, self.ssf_shift_q = init_ssf_scale_shift(attn_dim)
                self.ssf_scale_k, self.ssf_shift_k = init_ssf_scale_shift(attn_dim)
                self.ssf_scale_v, self.ssf_shift_v = init_ssf_scale_shift(attn_dim)
                self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(attn_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        if self.attn_adapter_name == 'pa':
            res = self.attn_adapter(x)

        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        
        if self.attn_adapter_name == 'ssf':
            q = ssf_ada(q, self.ssf_scale_q, self.ssf_shift_q)
            k = ssf_ada(k, self.ssf_scale_k, self.ssf_shift_k)
            v = ssf_ada(v, self.ssf_scale_v, self.ssf_shift_v)

        q = q.reshape(B, N, self.num_heads, self.attn_dim // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, self.attn_dim // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, self.attn_dim // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.attn_dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        if self.attn_adapter_name == 'pa':
            x = x + res
        if self.attn_adapter_name == 'ssf':
            x = ssf_ada(x, self.ssf_scale_1, self.ssf_shift_1)
        if self.attn_adapter_name == 'sa':
            x = self.attn_adapter(x)

        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, adapter=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, adapter=adapter)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, adapter=adapter)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.blk_adapter_name = None
        if adapter:
            self.blk_adapter_name = adapter.name
            if adapter.name == 'ssf':
                self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(dim)
                self.ssf_scale_2, self.ssf_shift_2 = init_ssf_scale_shift(dim)

    def forward(self, x):
        if self.blk_adapter_name == 'ssf':
            x = x + self.drop_path1(self.attn(ssf_ada(self.norm1(x), self.ssf_scale_1, self.ssf_shift_1)))
            x = x + self.drop_path2(self.mlp(ssf_ada(self.norm2(x), self.ssf_scale_2, self.ssf_shift_2)))
        else:
            x = x + self.drop_path1(self.attn(self.norm1(x)))
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, adapter=None):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.adapter = adapter
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        if adapter and adapter.name == 'ssf':
            self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)

        if self.adapter and self.adapter.name == 'ssf':
            x = ssf_ada(x, self.ssf_scale_1, self.ssf_shift_1)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., no_of_exits=12, norm_layer=nn.LayerNorm, accumulator=nn.Module, adapter=None, 
                 last_exit_only=False, blks_to_exit=None):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.depth = depth
        self.last_exit_only = last_exit_only

        if blks_to_exit is not None:
            self.blks_to_exit = blks_to_exit
        else:
            assert depth % no_of_exits == 0, 'depth % no of early exits must be = 0'
            no_of_blks_per_early_exit = depth // no_of_exits
            self.blks_to_exit = list(reversed(range(depth-1, -1, -1)[::no_of_blks_per_early_exit]))

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, adapter=adapter)
        num_patches = self.patch_embed.num_patches

        self.accumulator = accumulator
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, adapter=adapter)
            for i in range(self.depth)])
        # self.norm = norm_layer(embed_dim) # layernorm is used in each early exit mlp

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
            align_corners=False,
            recompute_scale_factor=False
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x):
        x = self.prepare_tokens(x)
        cls_tokens = []
        output_logits = []

        early_exit_idx = 0
        for blk_idx, blk in enumerate(self.blocks):
            x = blk(x)

            cls_token = x[:, 0][:, None, :]
            cls_tokens.append(cls_token)
            mod_tokens = None
            if blk_idx in self.blks_to_exit:
                if self.last_exit_only and blk_idx != self.blks_to_exit[-1]:
                    early_exit_idx += 1
                    # skip this exit
                    continue

                # classification
                exit_cls_tokens = torch.cat((cls_tokens), 1)

                if isinstance(self.accumulator, Ree):
                    mod_tokens = self.accumulator(exit_cls_tokens)
                    _outputs = self.accumulator.head(mod_tokens[:,0] + exit_cls_tokens[:, -1])
                elif isinstance(self.accumulator, nn.ModuleList): # layer-wise exit
                    _outputs = self.accumulator[early_exit_idx](exit_cls_tokens[:, -1]) # last class token
                    _outputs = _outputs.view(_outputs.shape[0], -1)
                else: 
                    _outputs = self.accumulator(exit_cls_tokens[:, -1]) # last class token
                    _outputs = _outputs.view(_outputs.shape[0], -1)
                output_logits.append(_outputs)

                early_exit_idx += 1
                
            # Replace the cls token that from accumulator
            if isinstance(self.accumulator, Ree): 
                if self.accumulator.modulation:
                    if mod_tokens is None:
                        mod_tokens = self.accumulator(torch.cat((cls_tokens), 1))
                    x[:, 0] = mod_tokens[:, -1]

        return output_logits


def vit_tiny(dim=192, patch_size=16, depth=12, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=dim, depth=depth, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), in_chans=3, **kwargs)
    return model


def vit_small(dim=384, patch_size=16, depth=12, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=dim, depth=depth, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), in_chans=3, **kwargs)
    return model


def vit_base(dim=768, patch_size=16, depth=12, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=dim, depth=depth, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), in_chans=3, **kwargs)
    return model

def reefl_vit_template(base_model, patch_size, depth, accumulator, adapter, 
        no_of_exits=None, 
        blks_to_exit=None, # explicitly set which blocks to exit (starting from 0) 
        freeze_base_model=True, 
        last_exit_only=False, 
        width_scale=1., # setting the width of the model
        device='cuda'):
    accumulator_fn = get_func_from_config(accumulator)
    # print(f'Creating VIT with adapter {adapter} and accumulator {accumulator["class"]}')

    assert adapter is None or adapter.name in ['lora', 'sa', 'pa', 'ssf']
    if last_exit_only:
        assert accumulator.layerwise

    if blks_to_exit is not None: # overwrite no_of_exits
        no_of_exits = len(blks_to_exit)

    if no_of_exits is None:
        no_of_exits = depth # default is one exit per blk

    if 'base' in base_model:
        dim = 768
    elif 'small' in base_model:
        dim = 384
    elif 'tiny' in base_model:
        dim = 192
    else:
        raise NotImplementedError()
    
    if width_scale is None:
        width_scale = 1.0 # backward compatibility
    assert width_scale > 0 and width_scale <= 1
    dim = int(dim * width_scale) # cut width

    if accumulator.layerwise:
        vit_accumulator = nn.ModuleList([accumulator_fn(dim=dim, base_model=base_model, adapter=adapter, **accumulator.args) for _ in range(no_of_exits)])
    else:
        vit_accumulator = accumulator_fn(dim=dim, base_model=base_model, adapter=adapter, **accumulator.args)

    if base_model == 'deit_base_patch16':
        model = vit_base
        url = "https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth"
    elif base_model == 'deit_small_patch16':
        model = vit_small
        url = "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth"
    elif base_model == 'deit_tiny_patch16':
        model = vit_tiny
        url = "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth"
        raise NotImplementedError()

    model = model(dim=dim,
                    patch_size=patch_size, 
                    depth=depth, 
                    no_of_exits=no_of_exits, 
                    accumulator=vit_accumulator, 
                    adapter=adapter,
                    last_exit_only=last_exit_only,
                    blks_to_exit=blks_to_exit)
    
    if os.path.exists(f'base_models/{base_model}.pt'):
        state_dict = torch.load(f'base_models/{base_model}.pt')
    else:
        state_dict = torch.hub.load_state_dict_from_url(url=url, model_dir="pretrained_model")["model"]
        print('Pretrained weights found at {}'.format(url))

        for k in ['head.weight', 'head.bias']:
            if k in state_dict:
                print(f"removing key {k} from pretrained checkpoint")
                del state_dict[k]
        os.makedirs(f'base_models', exist_ok=True)
        torch.save(state_dict, f'base_models/{base_model}.pt')
        
    # modify qkv to q, k, v
    for sd_k in list(state_dict.keys()):
        if 'qkv' in sd_k:
            q, k, v = torch.chunk(state_dict[sd_k], 3)
            state_dict[sd_k.replace('qkv', 'q')] = q
            state_dict[sd_k.replace('qkv', 'k')] = k
            state_dict[sd_k.replace('qkv', 'v')] = v
            del state_dict[sd_k]

    # loading of base model parameters
    if width_scale == 1:
        model.load_state_dict(state_dict, strict=False)
    else:
        param_idx = {}
        model_state_dict = model.state_dict()

        for k in model_state_dict.keys():
            param_idx[k] = [
                torch.arange(size) for size in model_state_dict[k].shape
            ]  

        model.load_state_dict(prune(state_dict, param_idx), strict=False)

    trainable_state_dict_keys = []
    all_state_dict_keys = [] # all trainable SD
    # filter out trainable parameters
    if not freeze_base_model:
        all_state_dict_keys = list(model.state_dict().keys())
        if not last_exit_only:
            trainable_state_dict_keys = list(model.state_dict().keys())
        else:
            # only train the last exit + the backbone model
            for sd_k in model.state_dict().keys():
                if 'accumulator' in sd_k:                    
                    if f'accumulator.{no_of_exits - 1}' in sd_k:
                        trainable_state_dict_keys.append(sd_k)
                else:
                    trainable_state_dict_keys.append(sd_k)

    else:
        for sd_k in model.state_dict().keys():
            if 'accumulator' in sd_k:
                if (last_exit_only and f'accumulator.{no_of_exits - 1}' in sd_k) or not last_exit_only:
                    trainable_state_dict_keys.append(sd_k)
                all_state_dict_keys.append(sd_k)
                continue
            
            # PEFT
            if adapter:
                if adapter.name == 'lora':
                    if 'ef_lora_A' in sd_k or 'ef_lora_B' in sd_k:
                        trainable_state_dict_keys.append(sd_k)
                        all_state_dict_keys.append(sd_k)
                elif adapter.name in ['sa', 'pa']:
                    if 'ffn_adapter' in sd_k or 'attn_adapter' in sd_k:
                        trainable_state_dict_keys.append(sd_k)
                        all_state_dict_keys.append(sd_k)
                elif adapter.name in ['ssf']:
                    if 'ssf_shift' in sd_k or 'ssf_scale' in sd_k:
                        trainable_state_dict_keys.append(sd_k)
                        all_state_dict_keys.append(sd_k)

    model.trainable_state_dict_keys = trainable_state_dict_keys
    model.all_state_dict_keys = all_state_dict_keys 
    if not last_exit_only:
        assert model.trainable_state_dict_keys == model.all_state_dict_keys

    return model



    