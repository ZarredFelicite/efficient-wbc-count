import torch.nn as nn
import torch.nn.utils.prune as prune


def prune_model(
    architecture, upper_module, last_module_name="last_module_name", amount=None
):
    if architecture == "mobilenet_v2":
        for module_name, module in upper_module.named_children():
            if module_name == "conv":
                length = len(list(module.children()))
                if length == 3:
                    prune.l1_unstructured(module[0][0], name="weight", amount=amount)
                    prune.remove(module[0][0], "weight")
                    prune.l1_unstructured(module[1], name="weight", amount=amount)
                    prune.remove(module[1], "weight")
                if length == 4:
                    prune.l1_unstructured(module[0][0], name="weight", amount=amount)
                    prune.remove(module[0][0], "weight")
                    prune.l1_unstructured(module[1][0], name="weight", amount=amount)
                    prune.remove(module[1][0], "weight")
                    prune.l1_unstructured(module[2], name="weight", amount=amount)
                    prune.remove(module[2], "weight")
            if module_name == "18":
                prune.l1_unstructured(module[0], name="weight", amount=amount)
                prune.remove(module[0], "weight")
            if module_name == "classifier":
                prune.l1_unstructured(module[1], name="weight", amount=amount)
                prune.remove(module[1], "weight")
            # if module_name == "0":
            #     prune.l1_unstructured(module[0], name="weight", amount=amount)
            #     prune.remove(module[0], "weight")
            else:
                last_module_name = module_name
                prune_model(architecture, module, last_module_name, amount=amount)

    elif architecture == "mobilenet_v3_small":
        for module_name, module in upper_module.named_children():
            length = len(list(module.children()))
            if length in [2, 3] and module_name != "block":
                prune.l1_unstructured(module[0], name="weight", amount=amount)
                prune.remove(module[0], "weight")
            elif length == 5 and module_name != "block":
                for module_name2, module2 in module.named_children():
                    if module_name2 in ["fc1", "fc2"]:
                        prune.l1_unstructured(module2, name="weight", amount=amount)
                        prune.remove(module2, "weight")
            elif module_name == "classifier":
                prune.l1_unstructured(module[0], name="weight", amount=amount)
                prune.remove(module[0], "weight")
                prune.l1_unstructured(module[3], name="weight", amount=amount)
                prune.remove(module[3], "weight")
            else:
                last_module_name = module_name
                prune_model(architecture, module, last_module_name, amount=amount)

    elif architecture == "regnet_x_400mf":
        for module_name, module in upper_module.named_children():
            if module_name == "0":
                prune.l1_unstructured(module, name="weight", amount=amount)
                prune.remove(module, "weight")
            else:
                prune_model(architecture, module, amount=amount)
    elif architecture == "shufflenet_v2_x1_0":
        for module_name, module in upper_module.named_children():
            if (("branch" in last_module_name) or ("conv" in last_module_name)) and (
                module_name == "0"
            ):
                prune.l1_unstructured(module, name="weight", amount=amount)
                prune.remove(module, "weight")
            else:
                last_module_name = module_name
                prune_model(architecture, module, last_module_name, amount=amount)
    elif architecture == "MIMO":
        for module_name, module in upper_module.named_children():
            if "conv" in module_name:
                prune.l1_unstructured(module, name="weight", amount=amount)
                prune.remove(module, "weight")
            else:
                last_module_name = module_name
                prune_model(architecture, module, last_module_name, amount=amount)
    else:
        for module_name, module in upper_module.named_children():
            if "conv" in module_name:
                # prune.ln_structured(module, name="weight", amount=0.4, n=2, dim=0)
                prune.l1_unstructured(module, name="weight", amount=amount)
                prune.remove(module, "weight")
            else:
                prune_model(architecture, module, amount=amount)


def prune_model2(model, amount=None):
    for module_name, module in model.named_children():
        if hasattr(module, "weight"):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")
        else:
            prune_model2(module, amount=amount)


def global_prune(model, amount=None):
    parameters_to_prune = []
    list_prune_layers(parameters_to_prune, model)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    remove_global_prune(model)


def list_prune_layers(parameters_to_prune, model):
    for module in model.children():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            parameters_to_prune.append((module, "weight"))
        else:
            list_prune_layers(parameters_to_prune, module)


def remove_global_prune(model):
    for module in model.children():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.remove(module, "weight")
        else:
            remove_global_prune(module)
