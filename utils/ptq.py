import torch
import torch.nn as nn
import torch.quantization
import torch.quantization.quantize_fx as quantize_fx
import torchvision


def fusion(upper_module_name, upper_module):
    for module_name, module in upper_module.named_children():
        if module_name == "conv1":
            # [print(x) for x, y in upper_module.named_children()]
            # print("-----------------")
            torch.quantization.fuse_modules(
                upper_module, [["conv1", "bn1", "relu1"]], inplace=True
            )
        elif module_name == "conv2":
            torch.quantization.fuse_modules(
                upper_module, [["conv2", "bn2", "relu2"]], inplace=True
            )
        else:
            fusion(module_name, module)


def post_training_quantize(model, device, cal_dataloader=None, cal_batches=100):
    a_observer = "MovingAverageMinMaxObserver"
    w_observer = "MovingAveragePerChannelMinMaxObserver"
    a_qscheme = "per_tensor_affine"
    w_qscheme = "per_channel_symmetric"
    qconfig = torch.quantization.QConfig(
        activation=getattr(torch.quantization, a_observer).with_args(
            dtype=torch.quint8, qscheme=getattr(torch, a_qscheme)
        ),
        weight=getattr(torch.quantization, w_observer).with_args(
            dtype=torch.qint8, qscheme=getattr(torch, w_qscheme)
        ),
    )
    # Insert quantization observers to calibrate correct quantization range and bias
    qconfig_dict = {"": qconfig}
    model = quantize_fx.prepare_fx(model.eval(), qconfig_dict)
    # Calibration
    model.to(device)
    for i, (x, _) in enumerate(cal_dataloader):
        out = model(x.to(device))
        if i == cal_batches:
            break
    model = quantize_fx.convert_fx(model.to("cpu"))
    return model


def multimodal_post_training_quantize(
    model, device, cal_dataloader=None, cal_batches=100
):
    a_observer = "MovingAverageMinMaxObserver"
    w_observer = "MovingAverageMinMaxObserver"
    a_qscheme = "per_tensor_affine"
    w_qscheme = "per_tensor_symmetric"
    qconfig = torch.quantization.QConfig(
        activation=getattr(torch.quantization, a_observer).with_args(
            dtype=torch.quint8, qscheme=getattr(torch, a_qscheme)
        ),
        weight=getattr(torch.quantization, w_observer).with_args(
            dtype=torch.qint8, qscheme=getattr(torch, w_qscheme)
        ),
    )
    # Insert quantization observers to calibrate correct quantization range and bias
    qconfig_dict = {"": qconfig}
    model = quantize_fx.prepare_fx(model.eval(), qconfig_dict)
    # Calibration
    model.to(device)
    for i, (Fluor_img, Green_img, Purple_img, Red_img, labels) in enumerate(
        cal_dataloader
    ):
        Fluor_img, Green_img, Purple_img, Red_img = (
            Fluor_img.to(device),
            Green_img.to(device),
            Purple_img.to(device),
            Red_img.to(device),
        )
        Fluor_preds, Green_preds, Purple_preds, Red_preds = model(
            Fluor_img, Green_img, Purple_img, Red_img
        )
        if i == cal_batches:
            break
    # Convert to fx graph mode quantized model
    model = quantize_fx.convert_fx(model.to("cpu"))
    return model


def fusion_shufflenet(upper_module_name, upper_module):
    for module_name, module in upper_module.named_children():
        if module_name == "conv1":
            torch.quantization.fuse_modules(module, [["0", "1", "2"]], inplace=True)
        elif module_name == "branch1":
            if len(list(module.named_children())) == 5:
                torch.quantization.fuse_modules(module, [["0", "1"]], inplace=True)
                torch.quantization.fuse_modules(module, [["2", "3", "4"]], inplace=True)
        elif module_name == "branch2":
            if len(list(module.named_children())) == 8:
                torch.quantization.fuse_modules(module, [["0", "1", "2"]], inplace=True)
                torch.quantization.fuse_modules(module, [["3", "4"]], inplace=True)
                torch.quantization.fuse_modules(module, [["5", "6", "7"]], inplace=True)
        elif module_name == "conv5":
            torch.quantization.fuse_modules(module, [["0", "1", "2"]], inplace=True)
        else:
            fusion(module_name, module)
