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
        fc6 = model.module.classifier[1:3](out)
        fc7 = model.module.classifier[3:5](fc6)
    return fc6, fc7

def cat_response(model, img_path):
    all_img = os.listdir(img_path)
    all_img_fc6 = torch.tensor([]).to(device)
    all_img_fc7 = torch.tensor([]).to(device)
    all_img_name = []
    for ii in all_img:
        xx6,xx7 = (ftest(img_path,ii,model))
        all_img_fc6 = torch.cat((all_img_fc6, xx6), 0)
        all_img_fc7 = torch.cat((all_img_fc7, xx7), 0)
        all_img_name.append(ii)
    return all_img_fc6,all_img_fc7,all_img_name

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
    heiti_response_fc6, heiti_response_fc7, all_heiti_name = cat_response(model_ft, heiti_path)

    try:
        deepdream_fc6_path = os.path.join(root_dir,'data','out-images',model_now[0:-4]+'_fc6')
        dp_fc6_response,useless_var,all_dp_fc6_name = cat_response(model_ft,deepdream_fc6_path)
    except:
        print('no fc6 img for model {}'.format(model_now))
        dp_fc6_response=torch.tensor([]).to(device)
        all_dp_fc6_name=[]
    try:
        deepdream_fc7_path = os.path.join(root_dir,'data','out-images',model_now[0:-4]+'_fc7')
        useless_var,dp_fc7_response,all_dp_fc7_name = cat_response(model_ft,deepdream_fc7_path)
    except:
        print('no fc7 img for model {}'.format(model_now))
        dp_fc7_response=torch.tensor([]).to(device)
        all_dp_fc7_name=[]

    dataNew = os.path.join(root_dir,'response','{}_response.mat'.format(model_now[8:-4]))
    scio.savemat(dataNew, {'heiti_response_fc6':heiti_response_fc6.cpu().numpy(), \
                            'heiti_response_fc7':heiti_response_fc7.cpu().numpy(), \
                            'all_heiti_name': all_heiti_name, \
                            'dp_fc6_response': dp_fc6_response.cpu().numpy(), \
                            'all_dp_fc6_name':all_dp_fc6_name, \
                            'dp_fc7_response': dp_fc7_response.cpu().numpy(), \
                            'all_dp_fc7_name': all_dp_fc7_name})

