# cs231n Assignment Solution

This repository contains the code, materials, and solution for Assignment of the cs231n 2019 Convolutional Neural Networks for Visual Recognition. http://cs231n.stanford.edu/2019/

## Assignment 1

In this assignment you will practice putting together a simple image classification pipeline, based on the k-Nearest Neighbor or the SVM/Softmax classifier. The goals of this assignment are as follows:

- Build basic Image Classification pipeline and the data-driven approach (train/predict stages)
- Create the train/val/test splits and the use of validation data for hyperparameter tuning.
- Write efficient vectorized code with numpy
- Implement and apply a k-Nearest Neighbor (kNN) classifier
- Implement and apply a Multiclass Support Vector Machine (SVM) classifier
- Implement and apply a Softmax classifier
- Implement and apply a Two layer neural network classifier
- Analyze the differences and tradeoffs between these classifiers
- Improve performance by using higher-level representations than raw pixels (e.g. color histograms, Histogram of Gradient (HOG) features)

#### [Q1: k-Nearest Neighbor classifier](assignment1/knn.ipynb)

#### [Q2: Training a Support Vector Machine](assignment1/svm.ipynb)

#### [Q3: Implement a Softmax classifier](assignment1/softmax.ipynb)

#### [Q4: Two-Layer Neural Network](assignment1/two_layer_net.ipynb)

#### [Q5: Higher Level Representations: Image Features](assignment1/features.ipynb)

## Assignment 2

In this assignment you will practice writing backpropagation code, and training Neural Networks and Convolutional Neural Networks. The goals of this assignment are as follows:

- understand Neural Networks and how they are arranged in layered architectures
- understand and be able to implement (vectorized) backpropagation
- implement various update rules used to optimize Neural Networks
- implement Batch Normalization and Layer Normalization for training deep networks
- implement Dropout to regularize networks
- understand the architecture of Convolutional Neural Networks and get practice with training these models on data
- gain experience with a major deep learning framework, such as TensorFlow or PyTorch.

#### [Q1: Fully-connected Neural Network](assignment2/FullyConnectedNets.ipynb)

#### [Q2: Batch Normalization](assignment2/BatchNormalization.ipynb)

#### [Q3: Dropout](assignment2/Dropout.ipynb)

#### [Q4: Convolutional Networks](assignment2/ConvolutionalNetworks.ipynb)

#### [Q5: PyTorch / TensorFlow on CIFAR-10](assignment2/TensorFlow.ipynb)

## Assignment 3

In this assignment you will implement recurrent networks, and apply them to image captioning on Microsoft COCO. You will also explore methods for visualizing the features of a pretrained model on ImageNet, and also this model to implement Style Transfer. Finally, you will train a Generative Adversarial Network to generate images that look like a training dataset!

The goals of this assignment are as follows:

- Understand the architecture of recurrent neural networks (RNNs) and how they operate on sequences by sharing weights over time
- Understand and implement both Vanilla RNNs and Long-Short Term Memory (LSTM) networks.
- Understand how to combine convolutional neural nets and recurrent nets to implement an image captioning system
- Explore various applications of image gradients, including saliency maps, fooling images, class visualizations.
- Understand and implement techniques for image style transfer.
- Understand how to train and implement a Generative Adversarial Network (GAN) to produce images that resemble samples from a dataset.

#### [Q1: Image Captioning with Vanilla RNNs ](assignment3/RNN_Captioning.ipynb)

#### [Q2: Image Captioning with LSTMs](assignment3/LSTM_Captioning.ipynb)

#### [Q3: Network Visualization: Saliency maps, Class Visualization, and Fooling Images](assignment3/NetworkVisualization-TensorFlow.ipynb)

#### [Q4: Style Transfer](assignment3/StyleTransfer-TensorFlow.ipynb)

#### [Q5: Generative Adversarial Networks](assignment3/Generative_Adversarial_Networks_TF.ipynb)
