import json
import os
import tempfile
from datetime import datetime
from test import evaluate as evaluate

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

# from effnetv2 import effnetv2_s
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm

import wandb
from utils import dataset
from utils.helper_functions import cmap_norm, create_model, visualise_data

matplotlib.use("Agg")
if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    data_dir = os.path.join(os.path.dirname(os.getcwd()), "data/classification")

    for architecture in config["architecture_list"]:
        for colour in ["Purple"]:  # ["Green", "Purple", "Red", "Fluor"]:
            kfold_acc, kfold_kappa, kfold_mcc, kfold_report = ([], [], [], [])
            kfold_cm = np.zeros((5, 5, 5), dtype=int)
            fold_list = ["0", "1", "2", "3", "4"]
            if config["load_stats"] and os.path.isfile(
                f"resume/{colour}_{architecture}_run_stats.json"
            ):
                with open(
                    f"resume/{colour}_{architecture}_run_stats.json", "r"
                ) as jfile:
                    stats = json.load(jfile)
                fold_list = fold_list[int(stats["last_fold"]) + 1 :]
                kfold_acc = stats["kfold_acc"]
                kfold_kappa = stats["kfold_kappa"]
                kfold_mcc = stats["kfold_mcc"]
                kfold_report = stats["kfold_report"]
                kfold_cm = np.array(stats["kfold_cm"])
            for fold in fold_list:
                # architecture = "efficientnet_v2_s"
                train_dataloader, val_dataloader = dataset.dataloader(
                    data_dir,
                    fold,
                    colour,
                    batch_size=config["batch_size"],
                    workers=config["workers"],
                    n=config["rand_aug"]["n"],
                    m=config["rand_aug"]["m"],
                )
                name = f"{colour}_{fold}_{architecture}"
                wandb.init(
                    project="Baseline Training Final",
                    group="Folds",
                    config=config,
                    name=name,
                    reinit=True,
                    tags=[fold, colour, architecture],
                    # settings=wandb.Settings(start_method="thread"),
                    # dir=tempfile.gettempdir(),
                    # resume=True,
                    # id="adoapabu",
                )
                save_path = os.path.join("models", name)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                model = create_model(architecture, "cuda", True, False)
                # model = effnetv2_s(num_classes=5)
                loss_fn = CrossEntropyLoss().to("cuda")
                scaler = torch.cuda.amp.GradScaler()
                optimizer = Adam(
                    model.parameters(), lr=config["lr"], weight_decay=0.0,  # 0.0005,
                )
                scheduler1 = lr_scheduler.OneCycleLR(
                    optimizer, max_lr=0.001, total_steps=4675, pct_start=0.1,
                )
                # scheduler2 = lr_scheduler.ReduceLROnPlateau(
                #     optimizer,
                #     mode="min",
                #     factor=0.1,
                #     patience=config["patience"],
                #     threshold_mode="abs",
                #     verbose=True,
                # )
                if wandb.run.resumed and os.path.isfile(f"{save_path}/checkpoint.pt"):
                    checkpoint = torch.load(f"{save_path}/checkpoint.pt")
                    model.load_state_dict(checkpoint["model_state_dict"])
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    start_epoch = checkpoint["epoch"] + 1
                else:
                    start_epoch = 0
                ## TRAINING
                model.train()
                device = config["device"]
                best_acc = 0
                val_loss = 0
                val_acc = 0
                train_loss = 0
                print(f"Training {name} from epoch {start_epoch}")
                for epoch in range(start_epoch, config["epochs"]):
                    epoch_loss = 0
                    train_loop = tqdm(
                        train_dataloader,
                        desc=f"E{epoch} Loss/Val:{train_loss:.2f}/{val_loss:.2f} Acc/Best:{val_acc*100:.1f}/{best_acc*100:.1f}",
                        postfix=str(datetime.now().time())[:5],
                    )
                    for i, (x, y) in enumerate(train_loop):
                        with torch.cuda.amp.autocast():
                            optimizer.zero_grad()
                            preds = model(x.to(device))
                            loss = loss_fn(preds, y.to(device))
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        epoch_loss += loss.item()
                        scheduler1.step()
                    train_loss = epoch_loss / len(train_dataloader)
                    # Validation
                    val_loss, val_acc, cm = evaluate(
                        model, device, val_dataloader, config
                    )[0:3]
                    # if epoch > 5:
                    #     scheduler2.step(val_loss)
                    lr = optimizer.state_dict()["param_groups"][0]["lr"]
                    wandb.log(
                        {
                            "Loss": {"Train": train_loss, "Val": val_loss},
                            "Accuracy": val_acc,
                            "Learning Rate": lr,
                        },
                        step=epoch,
                    )
                    # Model Saving
                    if epoch % 10 == 0:
                        plot = cmap_norm(cm)
                        plt.savefig(
                            f"{save_path}/{name}_{epoch}.png",
                            bbox_inches="tight",
                            dpi=100,
                        )
                        wandb.log(
                            {
                                "Confusion Matrix": wandb.Image(
                                    f"{save_path}/{name}_{epoch}.png"
                                )
                            },
                            step=epoch,
                        )
                        plt.close()
                        torch.save(
                            {
                                "epoch": epoch,
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "loss": train_loss,
                            },
                            f"{save_path}/checkpoint.pt",
                        )
                        wandb.save(f"{save_path}/checkpoint.pt")
                    if val_acc > best_acc:
                        best_acc = val_acc
                        torch.save(
                            model.state_dict(), os.path.join(save_path, "best_acc.pt"),
                        )
                        wandb.save(os.path.join(save_path, "best_acc.pt"), policy="end")
                print("Done Training")
                print(f"Loading best model with Acc: {best_acc:.3f}")
                model.load_state_dict(torch.load(save_path + "/best_acc.pt"))
                ## EVALUATION
                model.eval()
                # Model size
                # wandb.run.summary["Model Size"] = size_measure(model)
                val_loss, val_acc, cm, report, kappa, mcc = evaluate(
                    model, device, val_dataloader, config, 1, 1, 1
                )
                # sample_weight=torch.unique(torch.FloatTensor(test_data.targets),return_counts=True)[1].tolist()
                plot = cmap_norm(cm)
                plt.savefig(
                    f"{save_path}/{name}.png", bbox_inches="tight", dpi=200,
                )
                wandb.run.summary["Final CM"] = wandb.Image(f"{save_path}/{name}.png")
                plt.close()
                wandb.save(f"{save_path}/best_acc.pt")
                wandb.run.summary["Accuracy"] = val_acc
                wandb.run.summary["CM"] = cm
                wandb.run.summary["Report"] = report
                wandb.run.summary["Kappa"] = kappa
                wandb.run.summary["MCC"] = mcc
                kfold_acc.append(val_acc)
                kfold_kappa.append(kappa)
                kfold_mcc.append(mcc)
                kfold_report.append(report)
                kfold_cm[int(fold), :, :] = np.array(cm)
                stats = {
                    "last_fold": fold,
                    "kfold_acc": kfold_acc,
                    "kfold_kappa": kfold_kappa,
                    "kfold_mcc": kfold_mcc,
                    "kfold_report": kfold_report,
                    "kfold_cm": kfold_cm.tolist(),
                }
                with open(
                    f"resume/{colour}_{architecture}_run_stats.json", "w"
                ) as jfile:
                    json.dump(stats, jfile)
                wandb.finish()
            print("Folds complete")
            kfold_cm = np.mean(kfold_cm, axis=0, dtype=int)
            kfold_report = pd.json_normalize(kfold_report, sep="_")
            kfold_report = kfold_report.mean().to_dict()
            kfold_acc = sum(kfold_acc) / len(kfold_acc)
            kfold_kappa = sum(kfold_kappa) / len(kfold_kappa)
            kfold_mcc = sum(kfold_mcc) / len(kfold_mcc)
            wandb.init(
                project="Baseline Training Final",
                group="K-Fold",
                config=config,
                name=f"{colour}_{architecture}",
                reinit=True,
                tags=[colour, architecture, "K-Fold"],
                # dir=tempfile.gettempdir(),
            )
            wandb.log(
                {
                    "Accuracy": kfold_acc,
                    "Kappa": kfold_kappa,
                    "MCC": kfold_mcc,
                    "Report": kfold_report,
                    "CM": kfold_cm,
                }
            )
            wandb.finish()
