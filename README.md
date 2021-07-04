# Awesome Semantic Segmentation 
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
> ## List of awesome things around semantic segmentation :tada: 

Semantic segmentation is **a computer vision task in which we label specific regions of an image according to what's being shown**. Semantic segmentation awswers for the question: "*What's in this image, and where in the image is it located?*".

Semantic segmentation is a critical module in robotics related applications, especially autonomous driving, remote sensing. Most of the research on semantic segmentation is focused on improving the accuracy with less attention paid to computationally efficient solutions.

![Seft-driving-car](https://miro.medium.com/max/640/0*Q1cE-LVilYVDHjQc)

The recent appoarch in semantic segmentation is using deep neural network, specifically **Fully Convolutional Network** (a.k.a FCN). We can follow the trend of semantic segmenation approach at: [paper-with-code](https://paperswithcode.com/sota/semantic-segmentation-pascal-voc-2012).

Evaluate metrics: **mIOU**, accuracy, speed,...

## State-Of-The-Art (SOTA) methods of Semantic Segmentation
|                   | Paper                                                                            | Benchmark on PASALVOC12 | Release     | Implement                                                                                                                                                                                            |
|-------------------|-----------------------------------------------------------------------------------|-------------------------|-------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| EfficientNet-L2+NAS-FPN | [Rethinking Pre-training and Self-training](https://arxiv.org/pdf/2006.06882v2.pdf) | 90.5% | NeurIPS 2020   | [TF](https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/self_training) |
| DeepLab V3+       | [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/pdf/1802.02611v3.pdf) | 89%                     | ECCV 2018   | [TF](https://github.com/tensorflow/models/tree/master/research/deeplab), [Keras](https://github.com/bonlime/keras-deeplab-v3-plus), [Pytorch](https://github.com/jfzhang95/pytorch-deeplab-xception), [Demo](https://colab.sandbox.google.com/github/tensorflow/models/blob/master/research/deeplab/deeplab_demo.ipynb) |
| DeepLab V3        | [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/pdf/1706.05587v3.pdf)                     | 86.9%                   | 17 Jun 2017 | [TF](https://github.com/tensorflow/models/tree/master/research/deeplab), [TF](https://github.com/rishizek/tensorflow-deeplab-v3)                                                                                                                              |
| Smooth Network with Channel Attention Block        | [Learning a Discriminative Feature Network for Semantic Segmentation](https://arxiv.org/pdf/1804.09337v1.pdf)                     | 86.2%                   | CVPR 2018  | [Pytorch](https://github.com/ycszen/TorchSeg)                                                                                                                              |
| PSPNet            | [Pyramid Scene Parsing Network](https://arxiv.org/pdf/1612.01105v2.pdf)                                                     | 85.4%                   | CVPR 2017   | [Keras](https://github.com/hszhao/PSPNet), [Pytorch](https://github.com/warmspringwinds/pytorch-segmentation-detection), [Pytorch](https://github.com/kazuto1011/pspnet-pytorch)                     |
| ResNet-38 MS COCO | [Wider or Deeper: Revisiting the ResNet Model for Visual Recognition](https://arxiv.org/pdf/1611.10080v1.pdf)               | 84.9%                    | 30 Nov 2016 | [MXNet](https://github.com/itijyou/ademxapp)                                                                                                                                                         |
| RefineNet | [RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation](https://arxiv.org/abs/1611.06612)               | 84.2%                    | CVPR 2017 | [Matlab](https://github.com/guosheng/refinenet), [Keras](https://github.com/Attila94/refinenet-keras)                                                                                                                                                         |
| GCN | [Large Kernel Matters -- Improve Semantic Segmentation by Global Convolutional Network](https://arxiv.org/abs/1703.02719v1)               | 83.6%                    | CVPR 2017 | [TF](https://github.com/preritj/segmentation)                                                                                                                                                         |
| CRF-RNN | [Conditional Random Fields as Recurrent Neural Networks](https://arxiv.org/pdf/1502.03240v3.pdf)               | 74.7%                    | ICCV 2015 | [Matlab](https://github.com/torrvision/crfasrnn), [TF](https://github.com/sadeepj/crfasrnn_keras)                         |
| ParseNet | [ParseNet: Looking Wider to See Better](https://arxiv.org/pdf/1506.04579v2.pdf)               | 69.8%                    | 15 Jun 2015  | [Caffe](https://github.com/debidatta/caffe-parsenet)                         |
| Dilated Convolutions | [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/pdf/1511.07122v3.pdf)               | 67.6%                    | 23 Nov 2015  | [Caffe](https://github.com/fyu/dilation)                         |
| FCN | [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/pdf/1605.06211v1.pdf)               | 67.2%                    | CVPR 2015 | [Caffe](https://github.com/shelhamer/fcn.berkeleyvision.org)                         |
### Variants
* FCN with VGG(Resnet, Densenet) backbone: [pytorch](https://github.com/zengxianyu/FCN)
* The easiest implementation of fully convolutional networks (FCN8s VGG): [pytorch](https://github.com/pochih/FCN-pytorch)
* TernausNet (UNet model with VGG11 encoder pre-trained on Kaggle Carvana dataset [paper](https://arxiv.org/abs/1801.05746): [pytorch](https://github.com/ternaus/TernausNet)
* TernausNetV2: Fully Convolutional Network for Instance Segmentation: [pytorch](https://github.com/ternaus/TernausNetV2)


## Review list of Semantic Segmentation
* Evolution of Image Segmentation using Deep Convolutional Neural Network: A Survey 2020 ([University of Gour Banga,India](https://arxiv.org/pdf/2001.04074.pdf)) :star: :star: :star: :star: :star:
* A peek of Semantic Segmentation 2018 ([mc.ai](https://mc.ai/a-peek-at-semantic-segmentation-2018/)) :star: :star: :star: :star:
* Semantic Segmentation guide 2018 ([towardds](https://towardsdatascience.com/semantic-segmentation-with-deep-learning-a-guide-and-code-e52fc8958823)) :star: :star: :star: :star:
* An overview of semantic image segmentation ([jeremyjordan.me](https://www.jeremyjordan.me/semantic-segmentation/)) :star: :star: :star: :star: :star:
* Recent progress in semantic image segmentation 2018 ([arxiv](https://arxiv.org/abs/1809.10198), [towardsdatascience](https://towardsdatascience.com/paper-summary-recent-progress-in-semantic-image-segmentation-d7b93ee1b705)) :star: :star: :star: :star:
* A 2017 Guide to Semantic Segmentation Deep Learning Review ([blog.qure.ai](http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review#large-kernel)) :star: :star: :star: :star: :star:
* Review popular network architecture ([medium-towardds](https://towardsdatascience.com/@sh.tsang)) :star: :star: :star: :star: :star:
* Lecture 11 - Detection and Segmentation - CS231n ([slide](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture11.pdf), [vid](https://www.youtube.com/watch?v=nDPWywWRIRo)): :star: :star: :star: :star: :star: 
* A Survey of Semantic Segmentation 2016 ([arxiv](https://arxiv.org/pdf/1602.06541.pdf)) :star: :star: :star: :star: :star:

## Case studies
* Dstl Satellite Imagery Competition, 3rd Place Winners' Interview: Vladimir & Sergey: [Blog](http://blog.kaggle.com/2017/05/09/dstl-satellite-imagery-competition-3rd-place-winners-interview-vladimir-sergey/), [Code](https://github.com/ternaus/kaggle_dstl_submission)
* Carvana Image Masking Challenge–1st Place Winner's Interview: [Blog](http://blog.kaggle.com/2017/12/22/carvana-image-masking-first-place-interview/), [Code](https://github.com/asanakoy/kaggle_carvana_segmentation)
* Data Science Bowl 2017, Predicting Lung Cancer: Solution Write-up, Team Deep Breath: [Blog](http://blog.kaggle.com/2017/05/16/data-science-bowl-2017-predicting-lung-cancer-solution-write-up-team-deep-breath/)
* MICCAI 2017 Robotic Instrument Segmentation: [Code and explain](https://github.com/ternaus/robot-surgery-segmentation)
* 2018 Data Science Bowl Find the nuclei in divergent images to advance medical discovery: [1st place](https://www.kaggle.com/c/data-science-bowl-2018/discussion/54741), [2nd](https://www.kaggle.com/c/data-science-bowl-2018/discussion/61170), [3rd](https://www.kaggle.com/c/data-science-bowl-2018/discussion/56393), [4th](https://www.kaggle.com/c/data-science-bowl-2018/discussion/55118), [5th](https://www.kaggle.com/c/data-science-bowl-2018/discussion/56326), [10th](https://www.kaggle.com/c/data-science-bowl-2018/discussion/56238)
* Airbus Ship Detection Challenge: [4th place](https://www.kaggle.com/c/airbus-ship-detection/discussion/71667), [6th](https://www.kaggle.com/c/airbus-ship-detection/discussion/71782)

## Most used loss functions
* Pixel-wise cross entropy loss:
* Dice loss: which is pretty nice for balancing dataset
* Focal loss:
* Lovasz-Softmax loss:

## Datasets
* [Visual Object Classes Challenge 2012 (VOC2012)](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/): 400+ classes of real-world data
* [COCO Dataset](http://cocodataset.org/#home): 164k images, 72 classes: 80 thing classes, 91 stuff classes and 1 class 'unlabeled'
* [Cityscapes](https://www.cityscapes-dataset.com/): This dataset consists of segmentation ground truths for roads, lanes, vehicles and objects on road. The dataset contains 30 classes and of 50 cities collected over different environmental and weather conditions
* [PASCAL-Context](https://cs.stanford.edu/~roozbeh/pascal-context/)
* [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/): 20k+ images
* [Semantic3d](http://www.semantic3d.net/)
* [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)
* [lartpang/awesome-segmentation-saliency-dataset](https://github.com/lartpang/awesome-segmentation-saliency-dataset)
* [Kaggle](https://www.kaggle.com/search?q=segmentation+in%3Adatasets)

## Frameworks for segmentation
* [Semantic Segmentation in PyTorch](https://github.com/yassouali/pytorch_segmentation) (by yassouali): *Semantic segmentation models, datasets and losses implemented in PyTorch.*
* [Semantic Segmentation Suite](https://github.com/GeorgeSeif/Semantic-Segmentation-Suite) (by George Seif): *Semantic Segmentation Suite in TensorFlow. Implement, train, and test new Semantic Segmentation models easily!*
* [Segmentation Training Pipeline](https://github.com/petrochenko-pavel-a/segmentation_training_pipeline): *Research Pipeline for image masking/segmentation in Keras*
* [Tramac/awesome-semantic-segmentation-pytorch](https://github.com/Tramac/awesome-semantic-segmentation-pytorch) *Semantic Segmentation on PyTorch (include FCN, PSPNet, Deeplabv3, Deeplabv3+, DANet, DenseASPP, BiSeNet, EncNet, DUNet, ICNet, ENet, OCNet, CCNet, PSANet, CGNet, ESPNet, LEDNet, DFANet)*

## Related techniques 
* [Atrous/ Dilated Convolution](http://www.ee.bgu.ac.il/~rrtammy/DNN/StudentPresentations/TopazDCNN_CRF.pptx)
* [Transpose Convolution](https://towardsdatascience.com/up-sampling-with-transposed-convolution-9ae4f2df52d0) (Deconvolution, Upconvolution)
* [Unpooling](https://towardsdatascience.com/review-deconvnet-unpooling-layer-semantic-segmentation-55cf8a6e380e)
* [A technical report on convolution arithmetic in the context of deep learning](https://github.com/vdumoulin/conv_arithmetic)
* [CRF](https://arxiv.org/pdf/1711.04483.pdf)

> ## Feel free to show your :heart: by giving a star :star:

> ## :gift: [Check Out the List of Contributors](CONTRIBUTORS.md) - _Feel free to add your details here!_
