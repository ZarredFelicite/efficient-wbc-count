import argparse
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxruntime
import pandas as pd
import torch
import torch.nn as nn
import torch.onnx
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
)
from torch.profiler import ProfilerActivity, profile, record_function
from torchinfo import summary

import wandb

os.environ["WANDB_SILENT"] = "true"
from multimodal.test import run_MIMO
from multimodal.utils import models as mm_models
from multimodal.utils.metrics import computer_acc
from utils.dataset import dataloader, multi_cell_dataloader
from utils.helper_functions import cmap_norm, create_model
from utils.pruning import prune_model
from utils.ptq import fusion, multimodal_post_training_quantize, post_training_quantize
from utils.qat import mimo_quant_aware_training, quant_aware_training


def evaluate(model, device, loader, classes, report=False, kappa=False, mcc=False):
    batch_preds = []
    batch_labels = []
    batch_loss = 0
    model.eval()
    loss_fn = nn.CrossEntropyLoss().to(device)
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


def runtime_test(model, device, repetitions, batch_size=1, num_threads=0):
    # Measure Latency
    model.to(device)
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    elapsed = 0
    timings = np.zeros((repetitions, 1))
    dummy_input = torch.randn(batch_size, 3, 224, 224, dtype=torch.float).to(device)
    if num_threads != 0:
        torch.set_num_threads(num_threads)
    # GPU-WARM-UP
    for _ in range(10):
        _ = model(dummy_input)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            s = datetime.datetime.now()
            starter.record()
            _ = model(dummy_input)
            ender.record()
            elapsed += (datetime.datetime.now() - s).total_seconds() * 1000
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    time_cpu = elapsed / repetitions
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    return mean_syn, std_syn, time_cpu


def run_MIMO_latency(data, model, device="cpu", repetitions=50):
    model.to(device)
    elapsed = []
    with torch.no_grad():
        for i, (Fluor_img, Green_img, Purple_img, Red_img, labels) in enumerate(data):
            Fluor_img, Green_img, Purple_img, Red_img = (
                Fluor_img.to(device),
                Green_img.to(device),
                Purple_img.to(device),
                Red_img.to(device),
            )
            s = datetime.datetime.now()
            Fluor_preds, Green_preds, Purple_preds, Red_preds = model(
                Fluor_img, Green_img, Purple_img, Red_img
            )
            run_time = (datetime.datetime.now() - s).total_seconds() * 1000
            elapsed.append(run_time)
            if i == repetitions:
                break
        latency = sum(elapsed) / len(elapsed)
    return latency


def profile_inference(model, val_dataloader):
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        with record_function("model inference"):
            runtime = run_MIMO_latency(
                val_dataloader, model, device="cpu", repetitions=20
            )
    print(runtime)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))


def size_measure(model):
    torch.save(model.state_dict(), "temp.p")
    model_size = os.path.getsize("temp.p") / 1e6
    os.remove("temp.p")
    return model_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, default="fp32", help="Testing mode")
    parser.add_argument(
        "-f", "--folds", type=str, nargs="+", default=["0", "1", "2", "3", "4"]
    )
    parser.add_argument("-l", "--latency", action="store_true", help="Run latency test")
    parser.add_argument(
        "-b", "--model_size", action="store_true", help="Run model size test"
    )
    parser.add_argument(
        "-p", "--performance", action="store_true", help="Run performance test"
    )
    parser.add_argument(
        "-i", "--information", action="store_true", help="Get model summary"
    )
    parser.add_argument(
        "-w", "--wandb", type=int, help="Log to wandb: 0=None, 1=Final, 2=K-folds"
    )
    parser.add_argument(
        "-p", "--prune_amount", type=float, help="Amount to prune globally as fraction"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        mode = config["mode"]
        device = config["device"]

    for architecture in config["architecture_list"]:
        for colour in config["colours"]:
            print(f"Testing {architecture} with {colour} dataset")
            (
                kfold_acc,
                kfold_kappa,
                kfold_mcc,
                kfold_report,
                kfold_runtime,
                kfold_runtime_std,
                kfold_runtime_alt,
            ) = ([], [], [], [], [], [], [])
            num_folds = len(args.folds)
            kfold_cm = np.zeros((num_folds, 5, 5), dtype=int)
            for fold in args.folds:
                name = f"{colour}_{fold}_{architecture}"
                if architecture == "MIMO":
                    train_dataloader, val_dataloader = multi_cell_dataloader(
                        "data/classification", fold, batch_size=6, workers=6,
                    )
                    model = mm_models.MIMO_resnet34()
                    fp32model = f"models/multimodal/MIMO_folder_{fold}.pt"
                    model.load_state_dict(torch.load(fp32model, map_location="cuda"))
                else:
                    train_dataloader, val_dataloader = dataloader(
                        os.path.join(
                            os.path.dirname(os.getcwd()), "data/classification"
                        ),
                        fold,
                        colour,
                        batch_size=config["batch_size"],
                        workers=config["workers"],
                        n=20,
                        m=10,
                    )
                    model = create_model(
                        architecture, "cuda", pretrained=False, quantized=False
                    )
                    fp32model = os.path.join(
                        os.path.dirname(os.getcwd()), f"models/{name}/best_acc.pt"
                    )
                    model.load_state_dict(torch.load(fp32model, map_location="cuda"))
                wandb.init(
                    project="Final Testing",
                    config=config,
                    name=f"{mode}_{name}",
                    reinit=True,
                    tags=[fold, colour, architecture],
                    resume=False,
                    mode="online" if args.wandb == 2 else "disabled",
                )
                if args.mode == "ptq":
                    if architecture == "MIMO":
                        model = multimodal_post_training_quantize(
                            model,
                            "cuda",
                            cal_dataloader=train_dataloader,
                            cal_batches=50,
                        )
                        # save_path = f"models/{name}"
                        # if not os.path.exists(save_path):
                        #     os.makedirs(save_path)
                        # torch.jit.save(
                        #     torch.jit.script(model),^5K5c4&m
                        #     f"models/{name}/jit_scripted_ptq.pth",
                        # )
                    else:
                        model = post_training_quantize(
                            model,
                            "cuda",
                            cal_dataloader=train_dataloader,
                            cal_batches=50,
                        )
                        # torch.jit.save(
                        #     torch.jit.script(model),
                        #     f"models/{name}/jit_scripted_ptq.pth",
                        # )
                elif args.mode == "qat":
                    if architecture == "MIMO":
                        [
                            (Fluor_train_dataloader, _),
                            (Green_train_dataloader, _),
                            (Purple_train_dataloader, _),
                            (Red_train_dataloader, _),
                        ] = [
                            dataloader(
                                "classification",
                                fold,
                                colour,
                                config["batch_size"],
                                6,
                                n=0,
                                m=0,
                            )
                            for colour in ["Fluor", "Green", "Purple", "Red"]
                        ]
                        model = mimo_quant_aware_training(
                            model,
                            "cuda",
                            Fluor_train_dataloader,
                            Green_train_dataloader,
                            Purple_train_dataloader,
                            Red_train_dataloader,
                            val_dataloader,
                        )
                    else:
                        model = quant_aware_training(
                            model, "cuda", train_dataloader, val_dataloader,
                        )
                elif args.mode == "prune":
                    ## TODO option to prune alongside quantization
                    prune_model(architecture, model, amount=args.prune_amount)
                    model.to(device)
                model.eval()

                # Evaluation
                if args.performance:
                    if architecture == "MIMO":
                        pred_list, label_list, val_loss = run_MIMO(
                            data=val_dataloader, model=model, device="cpu"
                        )
                        cm, _, acc, _ = computer_acc(pred_list, label_list)
                        report = classification_report(
                            label_list,
                            pred_list,
                            labels=[4, 2, 3, 1, 0],
                            target_names=config["classes"],
                            sample_weight=None,
                            output_dict=True,
                            zero_division=0,
                        )
                    else:
                        loss, acc, cm, report, kappa, mcc = evaluate(
                            model,
                            "cuda" if (args.mode == "fp32") else "cpu",
                            val_dataloader,
                            config,
                            report=True,
                            kappa=True,
                            mcc=True,
                        )
                    if args.verbose:
                        print(f"Acc: {acc}")
                    kfold_acc.append(round(acc, 4))
                    kfold_report.append(report)
                    kfold_cm[int(fold), :, :] = np.array(cm)
                if args.latency:
                    if architecture == "MIMO":
                        runtime = run_MIMO_latency(
                            val_dataloader, model, device="cpu", repetitions=20
                        )
                    else:
                        runtime, runtime_std, runtime_alt = runtime_test(
                            model.eval(),
                            "cpu",
                            repetitions=20,
                            batch_size=8,
                            num_threads=0,
                        )
                    if args.verbose:
                        print(runtime)
                    kfold_runtime.append(round(runtime, 4))
                wandb.finish()
            if args.verbose:
                print("Folds complete")
            if args.model_size:
                model_size = size_measure(model)
            if args.information:
                ## TODO single modality
                if architecture == "MIMO":
                    model_summary = summary(
                        model,
                        (
                            (6, 3, 224, 224),
                            (6, 3, 224, 224),
                            (6, 3, 224, 224),
                            (6, 3, 224, 224),
                        ),
                    )
                else:
                    model_summary = summary(model, ((6, 3, 224, 224)))
                print(model_summary)

            name = f"{args.mode}_{colour}_{architecture}"
            if args.performance:
                kfold_cm = np.mean(kfold_cm, axis=0, dtype=int)
                plot = cmap_norm(cm)
                plt_save = f"results/plots/{mode}/{name}.png"
                plt.savefig(
                    plt_save, bbox_inches="tight", dpi=200,
                )
                plt.close()
                kfold_report = pd.json_normalize(kfold_report, sep="_")
                kfold_report = kfold_report.mean().to_dict()
                kfold_acc_mean = sum(kfold_acc) / len(kfold_acc)
            if args.latency:
                kfold_runtime_mean = sum(kfold_runtime) / len(kfold_runtime)
            wandb.init(
                project="Final Testing",
                config=config,
                name=name,
                reinit=True,
                tags=[colour, architecture, "K-Fold"],
                mode="online" if args.wandb else "disabled",
            )
            wandb.log(
                {
                    "Accuracy": kfold_acc_mean if args.performance else 0,
                    "Fold Accuracies": kfold_acc if args.performance else 0,
                    "Report": kfold_report if args.performance else 0,
                    "CM": kfold_cm if args.performance else 0,
                    "Final CM": wandb.Image(plt_save) if args.performance else 0,
                    "Runtime_mean": kfold_runtime_mean if args.latency else 0,
                    "Fold Runtimes": kfold_runtime if args.latency else 0,
                    "Model Size": model_size if args.model_size else 0,
                }
            )
            wandb.finish()
