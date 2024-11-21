from lib.model.dinod.modules.backbone.builder import build_backbone, build_preprocessing
from lib.model.dinod.modules.decoder.builder import build_decoder
from lib.model import ModelBuildingContext, ModelImplSuggestions
from .sample_data_generator import build_sample_input_data_generator

def get_DINOD_build_context(config: dict):
    print(f"DINOD model config: \n {config['MODEL']}")
    return build_DINOD_model(config)
    # return ModelBuildingContext(lambda impl_advice: build_DINOD_model(config, impl_advice),
    #                             lambda impl_advice: get_DINOD_build_string(config['MODEL']['TYPE'], impl_advice),
    #                             build_sample_input_data_generator(config))

def build_DINOD_model(config: dict): #, model_impl_suggestions: ModelImplSuggestions):
    model_config = config['MODEL']
    preprocess_config= model_config['PREPROCESS']
    common_config = config['COMMON']
    backbone = build_backbone(model_config['BACKBONE'],)
    decoder = build_decoder(model_config['DECODER'], )
                              # torch_jit_trace_compatible=model_impl_suggestions.torch_jit_trace_compatible)
    model_type = model_config['NAME']

    preprocessor, embeder = build_preprocessing(preprocess_config)

    if model_type == 'DINOD':
        # if model_impl_suggestions.optimize_for_inference:
        #     pass
        from .dinod import DINOD
        model = DINOD(preprocessor, embeder, backbone, decoder, **common_config)

    else:
        raise NotImplementedError(f"Model type '{model_type}' is not implemented.")

    return model

def get_DINOD_build_string(model_type: str, model_impl_suggestions: ModelImplSuggestions):
    build_string = 'DINOD'
    if 'full_finetune' in model_type:
        build_string += '_full_finetune'
    else:
        if model_impl_suggestions.optimize_for_inference:
            build_string += '_merged'
    if model_impl_suggestions.torch_jit_trace_compatible:
        build_string += '_disable_flash_attn'
    return build_string