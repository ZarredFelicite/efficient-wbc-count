import copy

import torch
import torch.quantization
import torch.quantization.quantize_fx as quantize_fx
import torchvision
import wandb
from multimodal.test import run_MIMO
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
)
from tqdm import tqdm
from tqdm.contrib import tzip


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


def quant_aware_training(model, device, train_dataloader, val_dataloader):
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
    model.train()
    qconfig_dict = {"": qconfig}
    model = quantize_fx.prepare_qat_fx(model, qconfig_dict)
    # model.qconfig = qconfig
    # fusion("model", model)
    # torch.quantization.prepare_qat(model, inplace=True)
    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    lr = 1e-10
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=0.1,
    )
    # optimizer = torch.optim.SGD(
    #     model.parameters(), lr=lr, momentum=0.0, weight_decay=0.1
    # )
    # lambda1 = lambda epoch: 0.65**epoch
    # scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer,
    #     lr_lambda=lambda1,
    # )
    best_acc = 0
    val_loss = 0
    val_acc = 0
    train_loss = 0
    classes = ["Basophil", "Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]
    for epoch in range(5):
        epoch_loss = 0
        train_loop = tqdm(
            train_dataloader,
            desc=f"E{epoch} Loss/Val:{train_loss:.2f}/{val_loss:.2f} Acc/Best:{val_acc*100:.1f}/{best_acc*100:.1f}",
        )
        for i, (x, y) in enumerate(train_loop):
            optimizer.zero_grad()
            preds = model(x.to(device))
            loss = loss_fn(preds, y.to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # quantized_model = quantize_fx.convert_fx(model.detach().to("cpu").eval())
            # quantized_model = torch.quantization.convert(
            #     model.to("cpu").eval(), inplace=False
            # )
            # model.to("cuda").train()
            if i % 50 == 0:
                quantized_model = copy.deepcopy(model).to("cpu").eval()
                quantized_model = quantize_fx.convert_fx(quantized_model)
                val_loss, val_acc, cm = evaluate(
                    quantized_model,
                    "cpu",
                    val_dataloader,
                    classes,
                )[0:3]
                wandb.log(
                    {"QAT Accuracy": val_acc}, step=(epoch * len(train_dataloader) + i)
                )
                if val_acc > best_acc:
                    best_acc = val_acc
        train_loss = epoch_loss / len(train_dataloader)
        if epoch > 3:
            model.apply(torch.quantization.disable_observer)
        if epoch > 2:
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
        val_loss, val_acc, cm = evaluate(
            model,
            "cuda",
            val_dataloader,
            classes,
        )[0:3]
        print(cm)

    print("Done Training")
    model = quantize_fx.convert_fx(model.to("cpu").eval())
    # quantized_model = torch.quantization.convert(model.to("cpu").eval(), inplace=False)
    return model.to("cpu").eval()


def mimo_quant_aware_training(
    model,
    device,
    Fluor_train_dataloader,
    Green_train_dataloader,
    Purple_train_dataloader,
    Red_train_dataloader,
    val_dataloader,
):
    model.to(device)
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
    qconfig_dict = {"": qconfig}
    model = quantize_fx.prepare_qat_fx(model, qconfig_dict)
    model.train()
    # Finetuning
    lr = 3e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)  # .0000005,
    for epoch in range(1):
        train_loss = 0
        loop = tzip(
            Fluor_train_dataloader,
            Green_train_dataloader,
            Purple_train_dataloader,
            Red_train_dataloader,
        )
        for i, data in enumerate(loop):
            Fluor_img = data[0][0].to(device)
            Green_img = data[1][0].to(device)
            Purple_img = data[2][0].to(device)
            Red_img = data[3][0].to(device)
            Fluor_label = data[0][1].to(device)
            Green_label = data[1][1].to(device)
            Purple_label = data[2][1].to(device)
            Red_label = data[3][1].to(device)
            Fluor_preds, Green_preds, Purple_preds, Red_preds = model(
                Fluor_img, Green_img, Purple_img, Red_img
            )
            loss = compute_MIMO_loss(
                Fluor_preds,
                Green_preds,
                Purple_preds,
                Red_preds,
                Fluor_label,
                Green_label,
                Purple_label,
                Red_label,
            )
            train_loss = train_loss + loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss = train_loss / len(Fluor_train_dataloader)
        if epoch > 3:
            model.apply(torch.quantization.disable_observer)
        if epoch > 2:
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
        pred_list, label_list, val_loss = run_MIMO(
            data=val_dataloader, model=model.eval(), device="cuda"
        )
        model.train()
        acc = accuracy_score(label_list, pred_list)
        print(
            f"Training Loss: {train_loss:.2f}     Val Loss: {val_loss:.2f}    Acc: {acc}"
        )
    print("Done Training")
    model.to("cpu").eval()
    # quantized_model = quantize_fx.fuse_fx(model)
    quantized_model = quantize_fx.convert_fx(model)
    quantized_model.eval()
    return quantized_model


def evaluate(model, device, loader, classes, report=False, kappa=False, mcc=False):
    batch_preds = []
    batch_labels = []
    batch_loss = 0
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    for i, (x, y) in enumerate(loader):
        preds = model(x.to(device))
        loss = loss_fn(preds, y.to(device))
        batch_loss += loss.item()
        preds = preds.max(1, keepdim=True)[1].cpu()
        preds = torch.squeeze(preds).tolist()
        labels = y.cpu().tolist()
        batch_preds.extend(preds)
        batch_labels.extend(labels)
    loss = batch_loss / len(loader)
    acc = accuracy_score(batch_labels, batch_preds)
    cm = confusion_matrix(batch_labels, batch_preds, labels=[4, 2, 3, 1, 0])
    if report:
        report = classification_report(
            batch_labels,
            batch_preds,
            labels=[0, 1, 2, 3, 4],
            target_names=classes,
            sample_weight=None,
            output_dict=True,
            zero_division=0,
        )
    if kappa:
        kappa = cohen_kappa_score(batch_labels, batch_preds)
    if mcc:
        mcc = matthews_corrcoef(batch_labels, batch_preds)
    return loss, acc, cm, report, kappa, mcc


def compute_MIMO_loss(
    Fluor_preds,
    Green_preds,
    Purple_preds,
    Red_preds,
    Fluor_labels,
    Green_labels,
    Purple_labels,
    Red_labels,
):
    loss_function = torch.nn.CrossEntropyLoss()
    Fluor_loss = loss_function(Fluor_preds, Fluor_labels)
    Green_loss = loss_function(Green_preds, Green_labels)
    Purple_loss = loss_function(Purple_preds, Purple_labels)
    Red_loss = loss_function(Red_preds, Red_labels)
    loss = Fluor_loss + Green_loss + Purple_loss + Red_loss
    return loss
