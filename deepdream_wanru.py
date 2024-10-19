# python script
# -*- coding: utf-8 LF
"""
@author: Wanru Li
@contact: wliwanru@gmail.com
@create: 2024/10/19-10:36
"""
"""
    This file contains the implementation of the DeepDream algorithm.

    If you have problems understanding any parts of the code,
    go ahead and experiment with functions in the playground.py file.
    
    --- wanru:
    run successfully on py38_cuda (3.8.20), torch 1.13.1+cu116 gtx1060
    
"""

import os
import argparse
import shutil
import time
import numpy as np
import torch
import cv2
from PIL import Image
import utils.utils as utils
from utils.constants import *
from tqdm import tqdm
import scipy.io as scio
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import h5py
from scipy import stats
from sklearn.decomposition import PCA


def resize_images(img_array, new_shape):
    """
    Resize a 4D array of images to a new shape.

    Parameters:
    img_array (numpy.ndarray): 4D array of images with shape (num_images, height, width, channels)
    new_shape (tuple): New shape for the images (new_height, new_width)

    Returns:
    numpy.ndarray: 4D array of resized images
    """
    height, width, channels, num_images = img_array.shape
    resized_images = np.zeros((new_shape[0], new_shape[1], channels, num_images), dtype=img_array.dtype)
    
    for i in range(num_images):
        resized_images[:, :, :, i] = cv2.resize(img_array[:, :, :, i], (new_shape[1], new_shape[0]))
    
    return resized_images


# loss.backward(layer) <- original implementation did it like this it's equivalent to MSE(reduction='sum')/2
def gradient_ascent(config, model, input_tensor, layer_ids_to_use, iteration):
    # Step 0: Feed forward pass
    # --- this only give a single value for fc6
    # --- that is only the activation of the selected channel ---
    out = model(input_tensor, config['channel'])  # --- input is a 3, 44, 44 degraded pyramid imgs for level 1
    
    # Step 1: Grab activations/feature maps of interest
    activations = [out[layer_id_to_use] for layer_id_to_use in layer_ids_to_use]
    
    # Step 2: Calculate loss over activations
    losses = []
    for layer_activation in activations:
        # Use torch.norm(torch.flatten(layer_activation), p) with p=2 for L2 loss and p=1 for L1 loss.
        # But I'll use the MSE as it works really good, I didn't notice any serious change when going to L1/L2.
        # using torch.zeros_like as if we wanted to make activations as small as possible but we'll do gradient ascent
        # and that will cause it to actually amplify whatever the network "sees" thus yielding the famous DeepDream look
        # loss_component = torch.nn.MSELoss(reduction='mean')(layer_activation, torch.zeros_like(layer_activation))
        loss_component = config['valance'] * layer_activation
        losses.append(loss_component)
    loss = torch.mean(torch.stack(losses))
    loss.backward()
    
    # Step 3: Process image gradients (smoothing + normalization)
    grad = input_tensor.grad.data
    
    # Applies 3 Gaussian kernels and thus "blurs" or smoothens the gradients and gives visually more pleasing results
    # sigma is calculated using an arbitrary heuristic feel free to experiment
    sigma = ((iteration + 1) / config['num_gradient_ascent_iterations']) * 2.0 + config['smoothing_coefficient']
    smooth_grad = utils.CascadeGaussianSmoothing(kernel_size=9, sigma=sigma)(grad)  # "magic number" 9 just works well
    
    # Normalize the gradients (make them have mean = 0 and std = 1)
    # I didn't notice any big difference normalizing the mean as well - feel free to experiment
    g_std = torch.std(smooth_grad)
    g_mean = torch.mean(smooth_grad)
    smooth_grad = smooth_grad - g_mean
    smooth_grad = smooth_grad / g_std
    
    # Step 4: Update image using the calculated gradients (gradient ascent step)
    input_tensor.data += config['lr'] * smooth_grad
    
    # Step 5: Clear gradients and clamp the data (otherwise values would explode to +- "infinity")
    input_tensor.grad.data.zero_()
    input_tensor.data = torch.max(torch.min(input_tensor, UPPER_IMAGE_BOUND), LOWER_IMAGE_BOUND)


def deep_dream_static_image(config, model):
    try:
        layer_ids_to_use = [model.layer_names.index(layer_name) for layer_name in config['layers_to_use']]
    except Exception as e:  # making sure you set the correct layer name for this specific model
        print(f'Invalid layer names {[layer_name for layer_name in config["layers_to_use"]]}.')
        print(f'Available layers for model {config["model_name"]} are {model.layer_names}.')
        return
    
    if config['use_noise']:
        shape = tuple([config['img_width'], config['img_width'], 3])
        img = np.random.uniform(low=0.0, high=1.0, size=shape).astype(np.float32)
    else:
        # load either the provided image or start from a pure noise image
        img_path = utils.parse_input_file(config['input'])
        # load a numpy, [0, 1] range, channel-last, RGB image
        img = utils.load_image(img_path, target_shape=config['img_width'])
    
    img = utils.pre_process_numpy_img(img)
    base_shape = img.shape[:-1]  # save initial height and width
    # Note: simply rescaling the whole result (and not only details, see original implementation) gave me better results
    # Going from smaller to bigger resolution (from pyramid top to bottom)
    for pyramid_level in range(config['pyramid_size']):
        new_shape = utils.get_new_shape(config, base_shape, pyramid_level)
        img = cv2.resize(img, (new_shape[1], new_shape[0]))
        input_tensor = utils.pytorch_input_adapter(img, DEVICE)
        for iteration in range(config['num_gradient_ascent_iterations']):
            h_shift, w_shift = np.random.randint(-config['spatial_shift_size'], config['spatial_shift_size'] + 1, 2)
            input_tensor = utils.random_circular_spatial_shift(input_tensor, h_shift, w_shift)
            gradient_ascent(config, model, input_tensor, layer_ids_to_use, iteration)
            input_tensor = utils.random_circular_spatial_shift(input_tensor, h_shift, w_shift, should_undo=True)
        img = utils.pytorch_output_adapter(input_tensor)
    return utils.post_process_numpy_img(img)


def ftest(pathnow, filename, model):
    img = Image.open(pathnow + "/" + filename).convert('RGB')
    img = data_transform(img).to(device)
    img = torch.unsqueeze(img, dim=0)
    model.eval()
    with torch.no_grad():
        out = model.features(img)
        out = model.avgpool(out)
        out = torch.flatten(out, 1)
        fc6 = model.classifier[1:3](out)
        fc7 = model.classifier[3:5](fc6)
    return fc6, fc7


def get_features_img_array(img_array, model):
    """
    Process a batch of images through the model and extract fc6 and fc7 features.

    Parameters:
    img_array (numpy.ndarray): Input array of images with shape (height, width, channels, n_imgs).
    model: Pre-trained model to use for feature extraction.

    Returns:
    numpy.ndarray: fc6 features with shape (n_imgs, fc6_features).
    numpy.ndarray: fc7 features with shape (n_imgs, fc7_features).
    """
    
    img_tensor_list = []
    for idx_img in range(img_array.shape[-1]):
        pil_img = Image.fromarray(img_array[:, :, :, idx_img].astype(np.uint8))
        transformed_img = data_transform(pil_img)
        img_tensor_list.append(transformed_img)
    
    # Stack all transformed images into a single tensor batch
    img_tensor = torch.stack(img_tensor_list).to(device)
    
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient computation for efficiency
        # Pass the batch through the feature extractor
        out = model.features(img_tensor)  # Input: (batch_size, 3, H, W) -> Output: (batch_size, C, H', W')
        
        # Average pooling and flattening
        out = model.avgpool(out)  # Output: (batch_size, C, 1, 1)
        out = torch.flatten(out, 1)  # Output: (batch_size, C)
        
        # Get activations from fc6 and fc7 layers
        fc6 = model.classifier[1:3](out)  # Apply part of the classifier for fc6
        fc7 = model.classifier[3:5](fc6)  # Apply part of the classifier for fc7
    
    # Convert the activations to NumPy arrays and return
    fc6 = fc6.cpu().numpy()
    fc7 = fc7.cpu().numpy()
    
    return fc6, fc7


def compute_rdm(data):
    num_samples = data.shape[0]
    rdm = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        for j in range(num_samples):
            rdm[i, j] = np.linalg.norm(data[i] - data[j])
    return rdm


sum_data_dir = 'Q:/data/project_data/Kofiko_AO/EphysSumData'
results_dir = f'{sum_data_dir}/encoding/deepdream/torch'

os.makedirs(results_dir, exist_ok=True)

# Only a small subset is exposed by design to avoid cluttering

# Replacing argparse with direct parameter setup
args = {
    "img_width": 224,  # Resize input image to this width
    "model_name": 'ALEXNET',  # Neural network (model) to use for dreaming
    "pyramid_ratio": 1.5,  # Ratio of image sizes in the pyramid
    "lr": 0.1,  # Learning rate for gradient ascent
    "should_display": False,  # Display intermediate results
    "spatial_shift_size": 32,  # Random pixel shift before gradient ascent
    "smoothing_coefficient": 0.5,  # Standard deviation for gradient smoothing
    "use_noise": True  # Use noise instead of input image
}

# Wrapping configuration into a dictionary
max_pyramid = 5
max_iteration = 200
config = dict()
for key, value in args.items():
    config[key] = value

# config['input_name'] = os.path.basename(config['input'])
config['use_noise'] = True
print('using ', DEVICE)

input_size = 224
device = torch.device("cuda")
data_transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load Model
model_ft = utils.give_me_pretrained_alexnet('models/ckpt_99_size1024.pth', DEVICE)
model_deep_dream = utils.fetch_and_prepare_model('ALEXNET', 'models/ckpt_99_size1024.pth', device)
fc7_weights = model_deep_dream.slice8[0].weight.data.cpu().numpy()

## Finished for VWFA， now for 657
Obj_657_data = h5py.File('M1_word_rsp.mat', 'r')['obj_1000_test'][()]
Word_657_data = h5py.File('M1_word_rsp.mat', 'r')['word_1000_test'][()]
data657 = np.concatenate([Word_657_data, Obj_657_data], axis=1)

subject_name = 'MaoDan'
recording_region = 'aLPP'
dataset_name = 'nsd1000'
n_neurons = 824
task_name = 'MixedAll'

neuron_data_fdir = os.path.join(sum_data_dir,
                                f'combined_all_resp_norm_extracted_{task_name}_{dataset_name}'
                                f'_resp_{subject_name}_aLPP_{n_neurons}neurons.mat')
img_array_fdir = os.path.join(sum_data_dir, f'img_array_{dataset_name}_original.mat')

with h5py.File(neuron_data_fdir, 'r') as f:
    neuron_data = f['data'][()].transpose()  # (n_imgs, n_neurons)
try:
    with h5py.File(img_array_fdir, 'r') as f:
        img_array = f['img_array'][()]
except Exception as e:
    print(f"Failed to read with h5py: {e}. Trying scipy.io.loadmat...")
    mat_data = scio.loadmat(img_array_fdir)
    img_array = mat_data['img_array']

img_array_resized = resize_images(img_array, (input_size, input_size))

n_neuron_pcs = 10
pca = PCA(n_components=n_neuron_pcs)
pca.fit(neuron_data)
reduced_neuron_data = pca.transform(neuron_data)

# fc6_activations = np.empty((0, 4096), dtype=np.float32)
# folder_path = 'data/WORD1000'
# for filename in tqdm(os.listdir(folder_path)):
#     img_path = os.path.join(folder_path, filename)
#     fc6, _ = ftest(folder_path, filename, model_ft)
#     fc6 = fc6.cpu().numpy()
#     fc6_activations = np.vstack((fc6_activations, fc6))

fc6_activations, _ = get_features_img_array(img_array_resized, model_ft)

n_activation_pcs = 50
pca = PCA(n_components=n_activation_pcs)
pca.fit(fc6_activations)
reduced_img_data = pca.transform(fc6_activations)
coefficients, residuals, _, _ = np.linalg.lstsq(reduced_img_data, reduced_neuron_data, rcond=None)
new_tr_PC = np.dot(pca.components_.transpose(), coefficients)
fc7_weights[0:10, :] = new_tr_PC.transpose()

model_deep_dream.slice8[0].weight.data = torch.from_numpy(fc7_weights).to(device)

config['layers_to_use'] = ['fc7']
config['pretrained_weights'] = 'Pretrained1024'
config['pyramid_size'] = max_pyramid
config['num_gradient_ascent_iterations'] = max_iteration
config['dump_dir'] = os.path.join(OUT_IMAGES_PATH, f'{config["pretrained_weights"][0:-4]}_{config["layers_to_use"][0]}')

for config['valance'] in [1, -1]:
    config['valance_name'] = 'pos' if config['valance'] == 1 else 'neg'
    valance_name = config['valance_name']
    
    img_array_deepdream = np.zeros((input_size, input_size, 3, 10), dtype=np.float32)
    img_array_corresponding = np.zeros((input_size, input_size, 3, 10), dtype=np.uint8)
    
    for config['channel'] in range(10):
        img = deep_dream_static_image(config,
                                      model_deep_dream)  # img=None -> will be loaded inside of deep_dream_static_image
        # temp_dump_path = utils.save_and_maybe_display_image(config, img)
        if (config['valance'] == 1):
            # --- for the certain neuronal component, find the highest response (to find corresponding img) ---
            img_index = np.argmax(reduced_neuron_data[:, config['channel']])
        else:
            img_index = np.argmin(reduced_neuron_data[:, config['channel']])
        
        img_corresponding = img_array_resized[:, :, :, img_index]
        
        img_array_deepdream[:, :, :, config['channel']] = img
        img_array_corresponding[:, :, :, config['channel']] = img_corresponding
    
    # Save the image arrays into a .mat file
    scio.savemat(os.path.join(results_dir, f'deepdream_results_{dataset_name}'
                                           f'_{subject_name}_aLPP_{n_neurons}neurons'
                                           f'_{n_neuron_pcs}celPc_{n_activation_pcs}fc6Pc_{valance_name}.mat'),
                 {
                     'img_array_deepdream': img_array_deepdream,
                     'img_array_corresponding': img_array_corresponding
                 })
    
    # Save each single image from the arrays to PNG files
    for channel in range(10):
        # Extract images for the current channel
        deepdream_img = img_array_deepdream[:, :, :, channel]
        corresponding_img = img_array_corresponding[:, :, :, channel]
        
        # Convert images to the correct format (0-255) for saving
        deepdream_img = (deepdream_img * 255).astype(np.uint8)
        corresponding_img = (corresponding_img).astype(np.uint8)
        
        # Create file names
        deepdream_filename = os.path.join(results_dir, f'deepdream_img_{dataset_name}'
                                                       f'_{subject_name}_aLPP_{n_neurons}neurons'
                                                       f'_{n_neuron_pcs}celPc_{n_activation_pcs}fc6Pc'
                                                       f'_{valance_name}_ch{channel + 1:02d}_gen.png')
        corresponding_filename = os.path.join(results_dir, f'deepdream_img_{dataset_name}'
                                                           f'_{subject_name}_aLPP_{n_neurons}neurons'
                                                           f'_{n_neuron_pcs}celPc_{n_activation_pcs}fc6Pc'
                                                           f'_{valance_name}_ch{channel + 1:02d}_max.png')
        
        # Save the images as PNG files
        Image.fromarray(deepdream_img).save(deepdream_filename)
        Image.fromarray(corresponding_img).save(corresponding_filename)
    
    print("Images and .mat file saved successfully.")
