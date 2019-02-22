# Awesome Semantic Segmentation
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
> ## List of awesome things around semantic segmentation :tada: 

Semantic segmentation is **a computer vision task in which we label specific regions of an image according to what's being shown**. Semantic segmentation awswers for the question: "*What's in this image, and where in the image is it located?*".

Semantic segmentation is a critical module in robotics related applications, especially autonomous driving, remote sensing. Most of the research on semantic segmentation is focused on improving the accuracy with less attention paid to computationally efficient solutions.

![Seft-driving-car](https://www.jeremyjordan.me/content/images/2018/05/deeplabcityscape.gif)

The recent appoarch in semantic segmentation is using deep neural network, specifically **Fully Convolutional Network** (a.k.a FCN). We can follow the trend of semantic segmenation approach at: [paper-with-code](https://paperswithcode.com/sota/semantic-segmentation-pascal-voc-2012).

## State-Of-The-Art (SOTA) methods of Semantic Segmentation
|                   | Papers                                                                            | Benchmark on PASALVOC12 | Release     | Implement                                                                                                                                                                                            |
|-------------------|-----------------------------------------------------------------------------------|-------------------------|-------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| DeepLab V3+       | Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation | 89%                     | ECCV 2018   | [TF](https://github.com/tensorflow/models/tree/master/research/deeplab), [Keras](https://github.com/bonlime/keras-deeplab-v3-plus), [Pytorch](https://github.com/jfzhang95/pytorch-deeplab-xception) |
| DeepLab V3        | Rethinking Atrous Convolution for Semantic Image Segmentation                     | 86.9%                   | 17 Jun 2017 | [TF](https://github.com/tensorflow/models/tree/master/research/deeplab)                                                                                                                              |
| PSPNet            | Pyramid Scene Parsing Network                                                     | 85.4%                   | CVPR 2017   | [Keras](https://github.com/hszhao/PSPNet), [Pytorch](https://github.com/warmspringwinds/pytorch-segmentation-detection), [Pytorch](https://github.com/kazuto1011/pspnet-pytorch)                     |
| ResNet-38 MS COCO | Wider or Deeper: Revisiting the ResNet Model for Visual Recognition               | 84.9                    | 30 Nov 2016 | [MXNet](https://github.com/itijyou/ademxapp)                                                                                                                                                         |
* DeepLabV3+:
* DeepLabV3:
* PSPNet:
* Large Kernel Matter:
* ResNet38 MSCOCO:
* Multipath-RefineNet:
* CRF-RNN:
* DeepLabV2:
* DeepLabV1:
* Fully Dilated Convolutions Neural Net:
* SegNet:
* FCN: 
## Review list of Semantic Segmentation
* A peek of Semantic Segmentation 2018 ([mc.ai](https://mc.ai/a-peek-at-semantic-segmentation-2018/))
* Recent progress in semantic image segmentation 2018 ([arxiv](https://arxiv.org/abs/1809.10198), [towardsdatascience](https://towardsdatascience.com/paper-summary-recent-progress-in-semantic-image-segmentation-d7b93ee1b705))
* A 2017 Guide to Semantic Segmentation Deep Learning Review ([blog.qure.ai](http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review#large-kernel))
## Most used loss function
* Pixel-wise cross entropy loss:
* Dice loss:
* Focal loss:

## Dataset

## Related work 

> ## Feel free to show your :heart: by giving a star :star:

> ## :gift: [Check Out the List of Contributors](CONTRIBUTORS.md) - _Feel free to add your details here!_
