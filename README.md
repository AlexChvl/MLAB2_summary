# MLAB2_summary
Collection of coding assignments from MLAB. These were initially completed in pair programming with other MLAB participants over three weeks in August 2022. I have since then reworked some aspects of the exercises.  Some files are reworked so that they can run on my laptop. Some files can only reasonably run on a gpu but are included here for reference.

I have removed any files from the course that I did not contribute to. This includes the instruction files and the test files. The test files included here were written by me. I have also removed course material that has to do with interpretability research in order to keep it private. All material here is freely available online.

The following details the contents of each file. "widj" should be read as "week i day j" (i = 0,1,2 and j = 1,2,3,4,5).

w1d2: Convolutional neural networks. We build convolutional neural networks from scratch using as_strided on Pytorch tensors. We use these to build our own custom Conv modules and we end by training a Resnet model on cifar10.

w1d3: Backpropagation: Built the backpropagation function in pytorch from class. Created a new Tensor class based on numpy arrays and wrote all basic automatic differentiation functions. Then build our own Module class and Linear modules, as in pytorch. In the end we are able to train a fully connected network on MNIST. 

w1d4: Part 1: Optimisation and hyperparameter search: We built our own optimizers (SGD, Adam and RMSProp). As a warmup to using neural nets, we trained a simple three layer fully connected neural network to memorise an arbitrary image. We decided to use Vermeer's Girl with the Pearl Earring. We break down the image into minibatches and optimise the network against the l1 loss. Once the model is trained, we evaluate it on the whole dataset (the image) and see that it fits the image well. We used the l1 loss and the l2 loss to compare the effect of the loss. The l1 loss gives reconstructed images which are noticeably more linear, and the l2 loss gives more detail.

w1d4: Part 2: Hyperparameter search: We train a resnet model on the cifar10 task. We use the resnet we built in w1d2 and the optimizer from w1d4 Part 1. We do some hyperparameter tuning and use Weights and Biases to track the progress.

w2d1: We build a Bert model from scratch, we load the weights from the HuggingFace transformer library and we use the model to make some next-token predictions.

w2d2: BERT day. 
Part 1: We download the imdb sentiment dataset in order to do next word prediction and sentiment prediction on the reviews. We inspect the dataset. In particular, we find that reviews tend to be either very positive or very negative. We also find that the distribution of number of words per review does not vary between positive and negative reviews. 

Part 2: We train a classifier based on BERT to predict the star rating on the IMDB review dataset. When running on cpus, we only train the model on 800 reviews (otherwise the task is unreasonably long).

Part 3: We set up a training procedure for BERT. 

Part 4: We train a small BERT model on a small amount of data.