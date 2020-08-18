import torch
from torch.autograd import Variable
import torch.nn.functional as F
import time
import os
import sys
import torch.nn as nn
import torchvision.transforms as transforms
import json
from mean import get_mean, get_std
from PIL import Image
import cv2
from datasets.ucf101 import load_annotation_data
from datasets.ucf101 import get_class_labels
from model import generate_model
from utils import AverageMeter
from opts import parse_opts
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from temporal_transforms import LoopPadding, TemporalRandomCrop
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose


def resume_model(opt, model):
    """ Resume model 
    """
    checkpoint = torch.load(opt.resume_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])


def predict(clip, model):
    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    spatial_transform = Compose([
        Scale((150, 150)),
        #Scale(int(opt.sample_size / opt.scale_in_test)),
        #CornerCrop(opt.sample_size, opt.crop_position_in_test),
        ToTensor(opt.norm_value), norm_method
    ])
    if spatial_transform is not None:
        # spatial_transform.randomize_parameters()
        clip = [spatial_transform(img) for img in clip]

    clip = torch.stack(clip, dim=0)
    clip = clip.unsqueeze(0)
    with torch.no_grad():
        print(clip.shape)
        outputs = model(clip)
        outputs = F.softmax(outputs)
    print(outputs)
    scores, idx = torch.topk(outputs, k=1)
    mask = scores > 0.6
    preds = idx[mask]
    return preds


if __name__ == "__main__":
    opt = parse_opts()
    print(opt)
    data = load_annotation_data(opt.annotation_path)
    class_to_idx = get_class_labels(data)
    device = torch.device("cpu")
    print(class_to_idx)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    model = generate_model(opt, device)

    # model = nn.DataParallel(model, device_ids=None)
    # print(model)
    if opt.resume_path:
        resume_model(opt, model)
        opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
        opt.std = get_std(opt.norm_value)
        model.eval()

        cam = cv2.VideoCapture(
            '/Users/pranoyr/Desktop/v_CricketShot_g11_c05.avi')
        clip = []
        frame_count = 0
        while True:
            ret, img = cam.read()
            if frame_count == 16:
                print(len(clip))
                preds = predict(clip, model)
                draw = img.copy()
                font = cv2.FONT_HERSHEY_SIMPLEX
                if preds.size(0) != 0:
                    print(idx_to_class[preds.item()])
                    cv2.putText(draw, idx_to_class[preds.item(
                    )], (100, 100), font, .5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.imshow('window', draw)
                    cv2.waitKey(1)
                frame_count = 0
                clip = []

            #img = cv2.resize(img, (224,224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #img = Image.fromarray(img.astype('uint8'), 'RGB')
            img = Image.fromarray(img)
            clip.append(img)
            frame_count += 1