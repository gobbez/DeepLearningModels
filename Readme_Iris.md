
# KERAS - IRIS DATASET

Train and test a Python Keras Deep Learning model to categorize different types of Iris.


## Iris Dataset

This data sets consists of 3 different types of irises’ (Setosa, Versicolour, and Virginica) petal and sepal length, stored in a 150x4 numpy.ndarray


## Deployment

You have to import iris dataset like this:

```bash
  from sklearn.datasets import load_iris
```

The dataset is load by this function:

```bash
  iris = load_iris()
```

The model then selects X and y and starts its training with 100 epochs.

After training phase it shows a plot with accuracies and saves the model in local.

## Conclusions
This model has a 100% accuracy and can categorize different types of Iris.

You are ready to use the model for other images or other training, too! 💥

## Documentation

[Iris Dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html)

