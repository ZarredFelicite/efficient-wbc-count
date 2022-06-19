import sys

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from aimet_common.defs import QuantScheme
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_torch.model_preparer import prepare_model
from aimet_torch.quantsim import QuantizationSimModel
from torch.utils.data import DataLoader as dataloader
from torchvision import models

sys.path.insert(1, "utils")
from HelperFunctions import *

os.environ["WANDB_SILENT"] = "true"
import wandb

wandb.login()
if __name__ == "__main__":
    device_name = torch.cuda.get_device_name(0)
    cuda_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu:0")
    print("Using cuda device: ", device_name)

    colour = "Purple"
    config_ptq = dict(
        classes=["Basophil", "Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"],
        image_size=160,
        batch_size=32,
        batch_size_latency=1,
        bit_width=16,
        save="Models/baseline/temp/" + colour,
        save_suffix="_" + colour,
        load_suffix="_profiling_" + colour,
        # test_set = colour + '_test',
        architecture_list=[
            "resnet18",
            "resnet34",
            "resnet50",
            # "regnet_y_400mf",
            "mobilenet_v3_small",
            "mobilenet_v2",
            # "squeezenet1_1",
            # "shufflenet_v2_x0_5",
            # "regnet_x_400mf",
            # "regnet_x_800mf",
            # "regnet_y_400mf",
            # "regnet_y_800mf",
        ],
        data_path="Data/WBC_Classification_3172_c/",
        logging="online",  #'disabled' #
    )
    transform_test = transforms.Compose([transforms.ToTensor()])
    test_data = datasets.ImageFolder(
        config_ptq["data_path"] + colour + "/test", transform=transform_test
    )
    print(f"Number of testing examples: {len(test_data)}")
    test_loader = dataloader(
        test_data, batch_size=config_ptq["batch_size"], pin_memory=True, drop_last=True,
    )

    for architecture in config_ptq["architecture_list"]:
        with wandb.init(
            project="AIMET",
            config=config_ptq,
            group="Baseline",
            tags=[colour, "Baseline", "aimet"],
            # job_type="eval",
            reinit=True,
            mode=config_ptq["logging"],
        ):
            wandb.config["architecture"] = architecture
            wandb.run.summary["architecture"] = architecture
            wandb.config["colour"] = colour
            config = wandb.config
            wandb.run.name = config["architecture"] + config["save_suffix"]
            float_model_file = (
                config_ptq["save"] + architecture + config_ptq["load_suffix"] + ".pt"
            )
            model = create_model(architecture, cuda_device, pretrained=False)
            check_point = torch.load(float_model_file)
            model.load_state_dict(check_point["model_state_dict"])
            model = prepare_model(model)
            model.to(cuda_device)
            model.eval()
            loss_fn = nn.CrossEntropyLoss().to(cuda_device)

            # results, cm = evaluate2(
            #     model=model,
            #     device=cuda_device,
            #     loader=test_loader,
            #     batch_size=config_ptq["batch_size"],
            #     loss_fun=loss_fn,
            #     labels_indx=list(test_data.class_to_idx.values()),
            #     names=list(test_data.class_to_idx.keys()),
            # )
            # # Log baseline results
            # wandb.run.summary["CM"] = cm
            # wandb.run.summary["Results"] = results
            # wandb.log(
            #     {
            #         "CM Plot": wandb.Image(
            #             return_cm(
            #                 cm,
            #                 config["classes"],
            #                 normalize=True,
            #                 title="Confusion matrix "
            #                 + config["architecture"]
            #                 + config["save_suffix"],
            #                 cmap=plt.cm.Blues,
            #             )
            #         )
            #     }
            # )
        #################
        # QUANTIZATION
        #################
        with wandb.init(
            project="AIMET",
            config=config_ptq,
            group="PTQ",
            tags=[colour, "PTQ", "aimet"],
            # job_type="eval",
            reinit=True,
            mode=config_ptq["logging"],
        ):
            wandb.config["architecture"] = architecture
            wandb.run.summary["architecture"] = architecture
            wandb.config["colour"] = colour
            config = wandb.config
            wandb.run.name = config["architecture"] + config["save_suffix"]
            # Folding batchnorm
            _ = fold_all_batch_norms(model, input_shapes=(1, 3, 160, 160))
            dummy_input = torch.rand(1, 3, 160, 160).cuda()
            # Inserting quantizers
            sim = QuantizationSimModel(
                model=model,
                quant_scheme=QuantScheme.post_training_tf_enhanced,
                dummy_input=dummy_input,
                default_output_bw=config["bit_width"],
                default_param_bw=config["bit_width"],
            )

            # Calibration
            def pass_calibration_data(sim_model, use_cuda):
                sim_model.eval()
                with torch.no_grad():
                    for x, y in test_loader:
                        x = x.to(cuda_device)
                        sim_model(x)

            sim.compute_encodings(
                forward_pass_callback=pass_calibration_data,
                forward_pass_callback_args=True,
            )

            # Evaluation
            results, cm = evaluate2(
                model=sim.model,
                device=cuda_device,
                loader=test_loader,
                batch_size=config_ptq["batch_size"],
                loss_fun=loss_fn,
                labels_indx=list(test_data.class_to_idx.values()),
                names=list(test_data.class_to_idx.keys()),
            )
            wandb.run.summary["CM"] = cm
            wandb.run.summary["Results"] = results
            # wandb.log(
            #     {
            #         "CM Plot": wandb.Image(
            #             return_cm(
            #                 cm,
            #                 config["classes"],
            #                 normalize=True,
            #                 title="Confusion matrix "
            #                 + config["architecture"]
            #                 + config["save_suffix"],
            #                 cmap=plt.cm.Blues,
            #             )
            #         )
            #     }
            # )

