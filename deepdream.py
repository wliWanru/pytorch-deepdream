"""
    This file contains the implementation of the DeepDream algorithm.

    If you have problems understanding any parts of the code,
    go ahead and experiment with functions in the playground.py file.
"""

import os
import argparse
import shutil
import time


import numpy as np
import torch
import cv2 as cv


import utils.utils as utils
from utils.constants import *
from tqdm import tqdm
import scipy.io as scio

# loss.backward(layer) <- original implementation did it like this it's equivalent to MSE(reduction='sum')/2
def gradient_ascent(config, model, input_tensor, layer_ids_to_use, iteration):
    # Step 0: Feed forward pass
    out = model(input_tensor,config['channel'])

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
        loss_component = config['valance']*layer_activation
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

    # load either the provided image or start from a pure noise image
    img_path = utils.parse_input_file(config['input'])
        # load a numpy, [0, 1] range, channel-last, RGB image
    img = utils.load_image(img_path, target_shape=config['img_width'])
    if config['use_noise']:
        shape = img.shape
        img = np.random.uniform(low=0.0, high=1.0, size=shape).astype(np.float32)

    img = utils.pre_process_numpy_img(img)
    base_shape = img.shape[:-1]  # save initial height and width
    # Note: simply rescaling the whole result (and not only details, see original implementation) gave me better results
    # Going from smaller to bigger resolution (from pyramid top to bottom)
    for pyramid_level in range(config['pyramid_size']):
        new_shape = utils.get_new_shape(config, base_shape, pyramid_level)
        img = cv.resize(img, (new_shape[1], new_shape[0]))
        input_tensor = utils.pytorch_input_adapter(img, DEVICE)
        for iteration in range(config['num_gradient_ascent_iterations']):
            h_shift, w_shift = np.random.randint(-config['spatial_shift_size'], config['spatial_shift_size'] + 1, 2)
            input_tensor = utils.random_circular_spatial_shift(input_tensor, h_shift, w_shift)
            gradient_ascent(config, model, input_tensor, layer_ids_to_use, iteration)
            input_tensor = utils.random_circular_spatial_shift(input_tensor, h_shift, w_shift, should_undo=True)
        img = utils.pytorch_output_adapter(input_tensor)
    return utils.post_process_numpy_img(img)


if __name__ == "__main__":

    # Only a small subset is exposed by design to avoid cluttering
    parser = argparse.ArgumentParser()
    # Common params
    parser.add_argument("--input", type=str, help="Input IMAGE or VIDEO name that will be used for dreaming", default='0.jpg')
    parser.add_argument("--img_width", type=int, help="Resize input image to this width", default=224)
    parser.add_argument("--model_name", choices=[m.name for m in SupportedModels],
                        help="Neural network (model) to use for dreaming", default=SupportedModels.ALEXNET.name)

    # Main params for experimentation (especially pyramid_size and pyramid_ratio)
    parser.add_argument("--pyramid_ratio", type=float, help="Ratio of image sizes in the pyramid", default=1.5)
    parser.add_argument("--lr", type=float, help="Learning rate i.e. step size in gradient ascent", default=0.1)
    # You usually won't need to change these as often
    parser.add_argument("--should_display", action='store_true', help="Display intermediate dreaming results (default False)")
    parser.add_argument("--spatial_shift_size", type=int, help='Number of pixels to randomly shift image before grad ascent', default=32)
    parser.add_argument("--smoothing_coefficient", type=float, help='Directly controls standard deviation for gradient smoothing', default=0.5)
    parser.add_argument("--use_noise", action='store_true', help="Use noise as a starting point instead of input image (default False)")
    args = parser.parse_args()

    # Wrapping configuration into a dictionary
    doing_fc7_feature = 1
    doing_fc6_word_unit = 1
    max_pyramid=5
    max_iteration = 201
    iteration_step = 10
    config = dict()
    for arg in vars(args):
        config[arg] = getattr(args, arg)
    config['input_name'] = os.path.basename(config['input'])
    config['use_noise'] = True
    print('using ', DEVICE)

    all_models = os.listdir(WEIGHT_DIR_PATH)

    global_start_time = time.time()
    if(doing_fc7_feature):
        config['layers_to_use']=['fc7']
        for model_now in all_models:
            start_time = time.time()
            config['pretrained_weights']=model_now
            config['dump_dir'] = os.path.join(OUT_IMAGES_PATH, f'{config["pretrained_weights"][0:-4]}_{config["layers_to_use"][0]}')
            print('Dreaming started for pc from ', config['pretrained_weights'])
            for config['pyramid_size'] in range(1,max_pyramid):
                for config['num_gradient_ascent_iterations'] in tqdm(range(10,max_iteration,iteration_step)):
                    for config['channel'] in range(5):
                        for config['valance'] in [1,-1]:
                            config['valance_name']='pos' if config['valance']==1 else 'neg'
                            model_ft = utils.fetch_and_prepare_model(config['model_name'], config['pretrained_weights'], DEVICE)
                            img = deep_dream_static_image(config, model_ft)  # img=None -> will be loaded inside of deep_dream_static_image
                            temp_dump_path = utils.save_and_maybe_display_image(config, img)
                            end_time = time.time()
                print('model ', config['pretrained_weights'][8:-4],' Psize ' ,config['pyramid_size'])
                print('Time ' ,end_time-start_time)

    if(doing_fc6_word_unit):
        config['valance']=1
        config['valance_name']='pos'
        config['layers_to_use']=['fc6']

        all_models = ['pretrained','exposure_100_0','exposure_100_25','exposure_100_50','exposure_100_75','exposure_100_100']
        for mm in all_models:
        
            model_now = 'alexnet_' + mm + '_pca.pth'
            selectivity_now = mm + '_dprime_idx_fc6.mat'
            interested_channel = scio.loadmat(os.path.join(SELECTIVITY_DIR_PATH,selectivity_now))['dprime_idx'][0][-5:]
            start_time = time.time()
            config['pretrained_weights']=model_now
            config['dump_dir'] = os.path.join(OUT_IMAGES_PATH, f'{config["pretrained_weights"][0:-4]}_{config["layers_to_use"][0]}')
            print('Dreaming started for fc6 unit from ', config['pretrained_weights'])
            for config['pyramid_size'] in range(1,max_pyramid):
                for config['num_gradient_ascent_iterations'] in tqdm(range(10,max_iteration,iteration_step)):
                    for config['channel'] in interested_channel:
                        model_ft = utils.fetch_and_prepare_model(config['model_name'], config['pretrained_weights'], DEVICE)
                        img = deep_dream_static_image(config, model_ft) 
                        temp_dump_path = utils.save_and_maybe_display_image(config, img)
                        end_time = time.time()
                print('model ', config['pretrained_weights'][8:-4],' Psize ' ,config['pyramid_size'])
                print('Time ' ,end_time-start_time)
    global_end_time = time.time()
    print('Global Time ' ,global_end_time-global_start_time)
