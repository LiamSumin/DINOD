def get_ema(model, ema_config):
    if ema_config.type =='ModelEMA':
        from .ModelEMA import ModelEMA
        ema = ModelEMA(model, **ema_config.CONFIG)

    else:
        raise NotImplementedError(f"EMA type '{ema_config.type} is not implemented")

    return ema