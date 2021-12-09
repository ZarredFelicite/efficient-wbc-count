import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as tF
# Saliency
from torchcam.methods import SmoothGradCAMpp, LayerCAM
from torchcam.utils import overlay_mask

#Global
classes=['Basophil','Eosinophil','Lymphocyte','Monocyte','Neutrophil']

def salience(model, img, layers=None, hue=None):
    #if no layers are specified, get all layers in model
    if layers==None:
        layers = [n for n,m in model.named_children() if 'layer' in n]
    #cam_extractor = SmoothGradCAMpp(model, target_layer='layer4')
    cam_extractor = LayerCAM(model, layers)
    #If image is a tensor, convert from uint8 to float
    if type(img).__name__ == 'Tensor':
        input = img.unsqueeze(0)/255
    else:
        raise TypeError('Not a tensor, add handling for np arrays and other types')
    out = model(input)
    print("{}% certainty the cell is a {}".format(int(torch.nn.functional.softmax(out, dim=1).max().item()*100), classes[out.squeeze(0).argmax().item()]))
    cams = cam_extractor(out.squeeze(0).argmax().item(), out)
    fused_cam = cam_extractor.fuse_cams(cams)
    #Visualize and plot raw cams and overlayed cams
    num = len(cam_extractor.target_names)
    #fig, axes = plt.subplots(2, num+1, figsize=(50,20))
    fig = plt.figure(constrained_layout=True, figsize=(30,10))
    ax1 = []
    ax2 = []
    gs = GridSpec(4,num+3 , figure=fig)
    ax1.append(fig.add_subplot(gs[:2, 1]))
    ax1.append(fig.add_subplot(gs[:2, 2]))
    ax1.append(fig.add_subplot(gs[:2, 3]))
    ax1.append(fig.add_subplot(gs[:2, 4]))
    ax1.append(fig.add_subplot(gs[:2, 5]))
    ax1.append(fig.add_subplot(gs[1:3, 0]))
    ax2.append(fig.add_subplot(gs[2:, 1]))
    ax2.append(fig.add_subplot(gs[2:, 2]))
    ax2.append(fig.add_subplot(gs[2:, 3]))
    ax2.append(fig.add_subplot(gs[2:, 4]))
    ax2.append(fig.add_subplot(gs[2:, 5]))
    if hue:
        #Distort colour
        ax1[num+1].imshow(tF.to_pil_image(tF.adjust_hue(img, hue_factor=hue))); ax1[num+1].axis('off'); ax1[num+1].set_title('Original')
        for idx, name, cam in zip(range(num), cam_extractor.target_names, cams):
            ax1[idx].imshow(cam.numpy()); ax1[idx].axis('off'); ax1[idx].set_title(name)
            overlay = overlay_mask(tF.to_pil_image(tF.adjust_hue(img, hue_factor=hue)), tF.to_pil_image(cam, mode='F'), alpha=0.5)
            ax2[idx].imshow(overlay); ax2[idx].axis('off'); ax2[idx].set_title(name)
        ax1[num].imshow(fused_cam.numpy()); ax1[num].axis('off'); ax1[num].set_title(" + ".join(cam_extractor.target_names))
        result = overlay_mask(tF.to_pil_image(tF.adjust_hue(img, hue_factor=hue)), tF.to_pil_image(fused_cam, mode='F'), alpha=0.5)
        ax2[num].imshow(result); ax2[num].axis('off'); ax2[num].set_title(" + ".join(cam_extractor.target_names))
        # Once you're finished, clear the hooks on your model
        cam_extractor.clear_hooks()
    else:
        ax1[num+1].imshow(tF.to_pil_image(img))
        for idx, name, cam in zip(range(num), cam_extractor.target_names, cams):
            ax1[idx].imshow(cam.numpy()); ax1[idx].axis('off'); ax1[idx].set_title(name)
            overlay = overlay_mask(tF.to_pil_image(img), tF.to_pil_image(cam, mode='F'), alpha=0.5)
            ax2[idx].imshow(overlay); ax2[idx].axis('off'); ax2[idx].set_title(name)
        ax1[num].imshow(fused_cam.numpy()); ax1[num].axis('off'); ax1[num].set_title(" + ".join(cam_extractor.target_names))
        result = overlay_mask(tF.to_pil_image(img), tF.to_pil_image(fused_cam, mode='F'), alpha=0.5)
        ax2[num].imshow(result); ax2[num].axis('off'); ax2[num].set_title(" + ".join(cam_extractor.target_names))
        # Once you're finished, clear the hooks on your model
        cam_extractor.clear_hooks()
    return fig