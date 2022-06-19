from utils.helper_functions import create_model
import torch
import os
import onnx
import onnxruntime
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
import PIL
import pandas as pd
from torchvision import transforms
import numpy as np


class WBC_DataReader(CalibrationDataReader):
    def __init__(self, csv_path, colour):
        self.csv_path = csv_path
        self.enum_data_dicts = []
        self.datasize = 0
        self.data_csv = pd.read_csv(csv_path)
        self.colour = colour
        self.transform = transforms.Compose([transforms.CenterCrop(224)])
        self.preprocess_flag = True

    def get_next(self):
        if self.preprocess_flag:
            self.preprocess_flag = False
            unconcatenated_batch_data = []
            for index in range(1):
                img_file = os.path.join(os.path.dirname(self.csv_path), self.data_csv[self.colour][index])
                img = PIL.Image.open(img_file)
                img = np.array(self.transform(img), dtype=np.float32)
                unconcatenated_batch_data.append(img)
            batch_data = np.expand_dims(np.concatenate(np.expand_dims(unconcatenated_batch_data, axis=0), axis=0), axis=0)
            batch_data = np.rollaxis(batch_data, 4, 2)
            self.datasize = len(batch_data)
            self.enum_data_dicts = iter([{'input': data} for data in batch_data])
        return next(self.enum_data_dicts, None)

        
def convert_onnx(model, save_path, img_size=224, quantize=False, DataReader=None):

    # img_size = 224
    # save_dir = os.path.join(data_model_path, 'models/onnx_models')
    # save_path = f"{save_dir}/{architecture}.onnx"
    # csv_path = f"{data_model_path}/data/classification/train_{fold}.csv"
    # dr = WBC_DataReader(csv_path, colour)

    x = torch.randn(1, 3, img_size, img_size, requires_grad=True)
    ## Test model outputs before and after
    # torch_out = model(x)
    # print(torch_out)
    # Export the model
    torch.onnx.export(model,              # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    save_path, # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=12,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output']) # the model's output names

    #Quantize
    int8_save_path = f"{save_path[:-5]}_int8.onnx"
    quantize_static(save_path,
                int8_save_path,
                DataReader)
                
    ## Export to ORT format
    # save_dir = os.path.join(data_model_path, 'models/onnx_models')
    # %cd ~
    # %cd {save_dir}
    # !{sys.executable} -m onnxruntime.tools.convert_onnx_models_to_ort --optimization_level basic ./ 