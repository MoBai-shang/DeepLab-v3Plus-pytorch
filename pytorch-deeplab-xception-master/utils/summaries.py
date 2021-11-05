import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from dataloaders.utils import decode_seg_map_sequence

class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def visualize_image(self, writer, dataset, image, target, output, global_step,display_num=12):

        '''
        display_num=min(display_num,image.shape[0])
        nrow=int(torch.sqrt(torch.tensor(display_num)))
        grid_image = make_grid(image[:display_num].clone().cpu().data, nrow=nrow, normalize=True)
        writer.add_image('Image', grid_image, global_step)

        grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:display_num], 1)[1].detach().cpu().numpy(),dataset=dataset),nrow=nrow, normalize=False, value_range=(0, 255))
        writer.add_image('Predicted label', grid_image, global_step)
        grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target[:display_num], 1).detach().cpu().numpy(),dataset=dataset), nrow=nrow, normalize=False, value_range=(0, 255))
        writer.add_image('Groundtruth label', grid_image, global_step)

        '''

        display_num = min(display_num, image.shape[0])
        nrow = int(torch.sqrt(torch.tensor(display_num * 3)))
        src = image[:display_num].clone().cpu().data
        predict = decode_seg_map_sequence(torch.max(output[:display_num], 1)[1].detach().cpu().numpy(), dataset=dataset)
        true = decode_seg_map_sequence(torch.squeeze(target[:display_num], 1).detach().cpu().numpy(), dataset=dataset)

        def norm_ip(img, low, high):
            img.clamp_(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))

        norm_ip(src, float(src.min()), float(src.max()))
        combine_imgs=torch.stack([src,predict,true],-3)#or dim=1
        shape=list(src.shape)
        shape[-2]=-1#or shape[0]=-1
        combine_imgs=combine_imgs.reshape(*shape)
        grid_image = make_grid(combine_imgs,nrow=nrow, normalize=False)
        writer.add_image('src-predict-truth label', grid_image, global_step)
