# MLAB2_summary
Collection of coding assignments from MLAB. These were initially completed in pair programming with other MLAB participants over three weeks in August 2022. Some material has been removed to keep it private.

w1d2: Convolutional neural networks. We build convolutional neural networks from scratch using as_strided on Pytorch tensors. We use these to build our own custom Conv modules and we end by training a Resnet model on cifar10.

w1d3: Backpropagation: Built the backpropagation function in pytorch from class. Created a new Tensor class based on numpy arrays and wrote all basic automatic differentiation functions. Then build our own Module class and Linear modules, as in pytorch. In the end we are able to train a fully connected network on MNIST. 

w1d4: Part 1: Optimisation and hyperparameter search: We built our own optimizers (SGD, Adam and RMSProp). As a warmup to using neural nets, we trained a simple three layer fully connected neural network to memorise an arbitrary image. We decided to use Vermeer's Girl with the Pearl Earring. We break down the image into minibatches and optimise the network against the l1 loss. Once the model is trained, we evaluate it on the whole dataset (the image) and see that it fits the image well. 

Part 2: We train a resnet model on the cifar10 task. We do some hyperparameter tuning and use Weights and Biases to track the progress.