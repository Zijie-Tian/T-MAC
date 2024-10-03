import numpy as np
import os
import tvm
from pathlib import Path
from tvm.autotvm.measure.measure_methods import request_remote
from t_mac.ops import QGeMMLUTBitsCodegen, QGeMMLUTBitsPreprocessorCodegen
from t_mac.weights import preprocess_weights
from t_mac.utils import get_default_device_kwargs, nmse
from t_mac.model_utils import extract_kernel_shapes, get_quantization_config, _Model
import logging

if __name__ == "__main__":
    # print(extract_kernel_shapes("hf-bitnet-3b", "/data/hf_models/bitnet_b1_58-3B"))
    _model = _Model(Path("/data/hf_models/bitnet_b1_58-3B"))
    
    
    
    hparams = _Model.load_hparams(Path("/data/hf_models/bitnet_b1_58-3B"))
    for name, data in _model.get_tensors():
        if name.endswith(".qweight"):
            print(name, "\t\t\t", data.shape)
            
    print(hparams)


