def get_model(args):
    model = None
    if args.model=='mlp_mixer':
        from mlp_mixer import MLPMixer
        model = MLPMixer(
            in_channels=3,
            img_size=args.size,
            hidden_size=args.hidden_size,
            patch_size = args.patch_size,
            hidden_c = args.hidden_c,
            hidden_s = args.hidden_s,
            num_layers = args.num_layers,
            num_classes=args.num_classes,
            drop_p=args.drop_p,
            off_act=args.off_act
        )
    else:
        raise ValueError(f"No such model: {args.model}")

    return model.to(args.device)
