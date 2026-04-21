
def get_net_builder(net_name, from_name: bool, peft_config=None, vit_config=None):
    """
    built network according to network name
    return **class** of backbone network (not instance).

    Args
        net_name: 'WideResNet' or network names in torchvision.models
        from_name: If True, net_buidler takes models in torch.vision models. Then, net_conf is ignored.
    """
    ######################################################################
    # assert net_name in ['timm/vit_base_patch16_224.augreg_in21k', 'timm/vit_base_patch14_reg4_dinov2.lvd142m', 'timm/vit_base_patch16_clip_224.openai', 'vit_small_patch2_32', 'timm/ViT-B-16-SigLIP', 'timm/vit_large_patch14_clip_224.openai', 'timm/vit_large_patch14_reg4_dinov2.lvd142m'], f"Model {net_name} is not supported. To be refactored."
    ######################################################################
    if from_name:
        import torchvision.models as nets
        model_name_list = sorted(name for name in nets.__dict__
                                 if name.islower() and not name.startswith("__")
                                 and callable(nets.__dict__[name]))

        if net_name not in model_name_list:
            assert Exception(f"[!] Networks\' Name is wrong, check net config, \
                               expected: {model_name_list}  \
                               received: {net_name}")
        else:
            return nets.__dict__[net_name]
    else:
        import src.nets as nets
        if net_name.startswith('timm/'):
            model_name = net_name.split('/')[1]
            def builder(*_args, **_kwargs):
                # _kwargs = {**_kwargs}
                # if args is not None and hasattr(args, 'img_size'):
                #     _kwargs['img_size'] = args.img_size
                # if args is not None and hasattr(args, 'peft_config'):
                #     _kwargs['peft_config'] = args.peft_config
                return nets.timm_builder(model_name, peft_config, vit_config, *_args, **_kwargs)
        else:
            builder = getattr(nets, net_name)
        return builder
