## Deep Dream :computer: + :ocean::zzz: = :heart:
This repo contains a PyTorch implementation of the Deep Dream algorithm (:link: blog by [Mordvintstev et al.](https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html)).
This repo is forked from https://github.com/gordicaleksa/pytorch-deepdream

Use this deepdream method to visualize the feature(PC) in FC7 of Alexnet

We used deepdream algorithm to generate images that maximize or minimize first several PCs in fc6 word space. We applied PCA to fc6 response of 1000 Heiti words and replaced the weights projecting from fc6 to fc7 with PCA coeffcient. Instead of performing gradient ascent on activity of the entire layer, we take the response of a single PC (or its negative value)  as the loss function. For each unit, 80 images were generated with pyramid size ranging from 1 to 4 and iterations ranging from 10 to 200. We further confirmed the validity of generated imagesc by projecting the resulting fc6 response pattern into word space. 
Deepdream code is available at https://github.com/liyipeng-moon/pytorch-deepdream
