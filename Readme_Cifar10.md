
# KERAS - CIFAR10 Deep Learning

Train and test a Python Keras Deep Learning model to categorize different images and use it on your own.


## Cifar-10 Dataset

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

Here are the classes in the dataset, as well as 10 random images from each: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


## Deployment

I have used Python Keras from Tensorflow. You can simply install tensorflow from pip or you can use this comand.

```bash
  pip install keras
```

The dataset is load by this function:

```bash
  keras.datasets.cifar10.load_data()
```

The model then selects X and y and starts its training with 100 epoch, but EarlyStopping is set and we can see that it stops after 25/26 epochs.

After training phase it shows a plot with accuracies and saves the model in local.

Then it asks user to continue and you can load an image to make it categorize it.

## Conclusions
This model works pretty well with an accuracy of more than 80%.
After training it saves the model and you can reuse the code bypassing the training-phase.

You are ready to use the model for other images or other training, too! 💥

## Documentation

[CIFAR10 Keras](https://keras.io/api/datasets/cifar10/)

