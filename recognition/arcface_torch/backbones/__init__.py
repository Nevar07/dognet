from .iresnet import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200
from .mobilefacenet import get_mbf
import torch
from collections import OrderedDict


def get_model(name, **kwargs):
    # resnet
    if name == "r18":
        return iresnet18(False, **kwargs)
    elif name == "r34":
        return iresnet34(False, **kwargs)
    elif name == "r50":
        return iresnet50(False, **kwargs)
    elif name == "r100":
        return iresnet100(False, **kwargs)
    elif name == "r200":
        return iresnet200(False, **kwargs)
    elif name == "r2060":
        from .iresnet2060 import iresnet2060
        return iresnet2060(False, **kwargs)

    elif name == "mbf":
        fp16 = kwargs.get("fp16", False)
        num_features = kwargs.get("num_features", 512)
        return get_mbf(fp16=fp16, num_features=num_features)

    elif name == "mbf_large":
        from .mobilefacenet import get_mbf_large
        fp16 = kwargs.get("fp16", False)
        num_features = kwargs.get("num_features", 512)
        return get_mbf_large(fp16=fp16, num_features=num_features)

    elif name == "vit_t":
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=256, depth=12,
            num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.1)

    elif name == "vit_t_dp005_mask0": # For WebFace42M
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=256, depth=12,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.0)

    elif name == "vit_s":
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=12,
            num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.1)
    
    elif name == "vit_s_dp005_mask_0":  # For WebFace42M
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=12,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.0)
    
    elif name == "vit_b":
        # this is a feature
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=24,
            num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.1, using_checkpoint=True)

    elif name == "vit_b_dp005_mask_005":  # For WebFace42M
        # this is a feature
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=24,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.05, using_checkpoint=True)

    elif name == "vit_l_dp005_mask_005":  # For WebFace42M
        # this is a feature
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=768, depth=24,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.05, using_checkpoint=True)

    elif name == 'conv_base':
        from torchvision import models
        import torch.nn as nn
        n_features = kwargs.get("num_features", 512)
        model = models.convnext_base(pretrained=False)

        num_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_features,6000)

        ckpt = torch.load('./pretrain/output/cb_1/best_model.pt')
        # new_ckpt = OrderedDict()
        # for k,v in ckpt.items():
        #     name = k[7:]
        #     new_ckpt[name] = v
        # ckpt = new_ckpt
        model.load_state_dict(ckpt)

        model.classifier[2] = nn.Linear(num_features, n_features)
        return model

    elif name == 'conv_tiny':
        from torchvision import models
        import torch.nn as nn
        n_features = kwargs.get("num_features", 512)
        model = models.convnext_tiny(pretrained=False)

        num_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_features,6000)

        ckpt = torch.load('./pretrain/output/ctpt/best_model.pt43')
        # new_ckpt = OrderedDict()
        # for k,v in ckpt.items():
        #     name = k[7:]
        #     new_ckpt[name] = v
        # ckpt = new_ckpt
        model.load_state_dict(ckpt)

        model.classifier[2] = nn.Linear(num_features, n_features)
        return model
    

    elif name == 'vitb32':
        from torchvision import models
        import torch.nn as nn
        n_features = kwargs.get("num_features", 512)

        model = models.vit_b_32(pretrained=True)
        num_features = model.heads[0].in_features
        model.heads[0] = nn.Linear(num_features, 6000)

        ckpt = torch.load('./pretrain/output/vitb32_4/best_model.pt134')
        model.load_state_dict(ckpt)

        model.heads[0] = nn.Linear(num_features, n_features)

        return model

    elif name == 'swtb':
        import torch.nn as nn
        from pretrain.src.swt import build_model

        n_features = kwargs.get("num_features", 512)

        model = build_model('swin')
        num_features = model.head.in_features
        model.head = nn.Linear(num_features, 6000)

        ckpt = torch.load('./pretrain/output/swtb_1/best_model.pt')
        model.load_state_dict(ckpt)

        model.head = nn.Linear(num_features, n_features)

        return model


    else:
        raise ValueError()
