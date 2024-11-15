import copy
import torch

def _build_decoder(decoder_config, load_pretrained):
    decoder_config= copy.deepcopy(decoder_config)

    if decoder_config['TYPE'] == 'RTDETR_DECODER':
        decoder_config = decoder_config['CONFIG']
        from .rtdetr_decoder import RTDETRTransformer
        decoder = RTDETRTransformer(**decoder_config)
    else :
        raise Exception(f"unsupported {decoder_config['type']}")

    return decoder


def build_decoder(decoder_config: dict, load_pretrained=False):
    return _build_decoder(decoder_config, load_pretrained)