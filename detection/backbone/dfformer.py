from functools import partial

from mmdet.models.builder import BACKBONES
from torch import nn

from models.dfformer import DynamicFilter, MetaFormer, load_weights, default_cfgs


@BACKBONES.register_module()
class DetDFFormerS18(MetaFormer):
    def __init__(self, **kwargs):
        depths = [3, 3, 9, 3]
        dims = [64, 128, 320, 512]
        token_mixers = partial(DynamicFilter, weight_resize=True)
        head_fn = nn.Identity
        input_size = (3, 800, 800)
        super().__init__(
            depths=depths, dims=dims,
            token_mixers=token_mixers, head_fn=head_fn,
            input_size=input_size,
            fork_feat=True,
            **kwargs)
        self.default_cfg = default_cfgs['dfformer_s18']
        load_weights(self, input_size)


@BACKBONES.register_module()
class DetDFFormerS36(MetaFormer):
    def __init__(self, **kwargs):
        depths = [3, 12, 18, 3]
        dims = [64, 128, 320, 512]
        token_mixers = partial(DynamicFilter, weight_resize=True)
        head_fn = nn.Identity
        input_size = (3, 800, 800)
        super().__init__(
            depths=depths, dims=dims,
            token_mixers=token_mixers, head_fn=head_fn,
            input_size=input_size,
            fork_feat=True,
            **kwargs)
        self.default_cfg = default_cfgs['dfformer_s36']
        load_weights(self, input_size)


@BACKBONES.register_module()
class DetDFFormerM36(MetaFormer):
    def __init__(self, **kwargs):
        depths = [3, 12, 18, 3]
        dims = [96, 192, 384, 576]
        token_mixers = partial(DynamicFilter, weight_resize=True)
        head_fn = nn.Identity
        input_size = (3, 800, 800)
        super().__init__(
            depths=depths, dims=dims,
            token_mixers=token_mixers, head_fn=head_fn,
            input_size=input_size,
            fork_feat=True,
            **kwargs)
        self.default_cfg = default_cfgs['dfformer_m36']
        load_weights(self, input_size)
