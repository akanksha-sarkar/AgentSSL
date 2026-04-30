
def get_net_builder(net_name,  peft_config=None, vit_config={"drop_path_rate": 0.0} ):
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
    assert net_name in ['timm/vit_base_patch16_clip_224.openai', 'timm/vit_base_patch14_reg4_dinov2.lvd142m'], f"Model {net_name} is not supported. To be refactored."
    import nets
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
