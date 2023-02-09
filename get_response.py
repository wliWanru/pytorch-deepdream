from psutil import boot_time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torch.utils.data import sampler
from torchvision import datasets, models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd
from tqdm import tqdm
import scipy.io as scio
import time


input_size = 224
device = torch.device("cuda")
data_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
def ftest(pathnow,filename,model):
    img = Image.open(pathnow+ "/" + filename).convert('RGB')
    img = data_transform(img).to(device)
    img = torch.unsqueeze(img, dim=0)
    model.eval()
    with torch.no_grad():
        out = model.module.features(img)
        out = model.module.avgpool(out)
        out = torch.flatten(out, 1)
        fc7 = model.module.classifier[0:5](out)
    return fc7

root_dir = 'C:\\Users\\DELL\\Documents\\GitHub\\pytorch-deepdream'

all_models = os.listdir(os.path.join(root_dir,'weights'))
for model_now in all_models:
    model_ft = models.alexnet(weights=None)
    model_ft.to(device)
    model_ft.eval()
    state_dict = torch.load(root_dir + '/weights/' + model_now)

    new_state_dict = {}  # modify key names and make it compatible with current PyTorch model naming scheme
    for old_key in state_dict.keys():
        new_key = old_key.replace('.module', '')
        new_state_dict[new_key] = state_dict[old_key]
    model_ft = torch.nn.DataParallel(model_ft)
    model_ft.module.classifier[-1] = torch.nn.Linear(model_ft.module.classifier[-1].in_features,new_state_dict['module.classifier.6.bias'].shape[0])
    model_ft.load_state_dict(new_state_dict, strict=True)

    heiti_path = os.path.join(root_dir,'data','heiti')
    heiti_image = os.listdir(heiti_path)
    all_heiti_data = torch.tensor([]).to(device)
    all_heiti_name = []
    for ii in heiti_image:
        xx = (ftest(heiti_path,ii,model_ft))
        all_heiti_data = torch.cat((all_heiti_data, xx), 0)
        all_heiti_name.append(ii)


    deepdream_path = os.path.join(root_dir,'data','out-images',model_now[0:-4])
    deepdream_image = os.listdir(deepdream_path)
    all_dp_data = torch.tensor([]).to(device)
    all_dp_name = []
    for ii in deepdream_image:
        xx = (ftest(deepdream_path,ii,model_ft))
        all_dp_data = torch.cat((all_dp_data, xx), 0)
        all_dp_name.append(ii)

    dataNew = os.path.join(root_dir,'response','{}_response.mat'.format(model_now[8:-4]))
    scio.savemat(dataNew, {'all_heiti_data': all_heiti_data.cpu().numpy(), 'all_heiti_name': all_heiti_name,'all_dp_data': all_dp_data.cpu().numpy(), 'all_dp_name': all_dp_name})

