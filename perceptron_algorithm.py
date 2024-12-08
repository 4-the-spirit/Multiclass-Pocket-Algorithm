import itertools
import numpy as np


class PerceptronAlgorithm:
  """
    Implementation of the Perceptron Learning Algorithm.

    This class handles binary classification tasks using the perceptron algorithm.
    The labels of the training data must be either 1 or -1.

    Attributes:
        POSITIVE_CLASS_VALUE (int): The label representing the positive class (+1).
        NEGATIVE_CLASS_VALUE (int): The label representing the negative class (-1).
        training_data (numpy.ndarray): Feature vectors of the training examples.
        training_labels (numpy.ndarray): Labels corresponding to the training examples.
        weights_vec (numpy.ndarray): The weight vector for the perceptron model.
        b (float): The bias term for the perceptron model.
  """
  POSITIVE_CLASS_VALUE = 1
  NEGATIVE_CLASS_VALUE = -1

  def __init__(self, training_data, training_labels):
    self.training_data = training_data
    self.training_labels = training_labels
    self.weights_vec = np.zeros(len(self.training_data[0]))
    self.b = 0
    self.__error_data = None

  def from_parameters(self, weights_vec, b):
    """
        Creates a new perceptron instance with specified weights and bias.

        Args:
            weights_vec (numpy.ndarray): The weight vector to initialize the perceptron.
            b (float): The bias term to initialize the perceptron.

        Returns:
            PerceptronAlgorithm: A new perceptron instance with the specified parameters.
    """
    perceptron = PerceptronAlgorithm(self.training_data, self.training_labels)
    perceptron.weights_vec = weights_vec
    perceptron.b = b
    return perceptron

  def run(self, updates, max_epochs=1):
    """
        Trains the perceptron using the training data.

        Args:
            updates (int): The maximum number of updates to perform on the weights.
            max_epochs (int): The maximum number of passes through the dataset.

        Returns:
            None
    """
    updates_count = 0
    epochs_count = 0
    for i in itertools.cycle(range(len(self.training_data))):
      if i == len(self.training_data) - 1:
        epochs_count += 1
      if updates_count >= updates or epochs_count >= max_epochs:
        break

      actual_label = self.training_labels[i]
      predicted_label = self.classify(self.training_data[i])
      # Update the weights only if the predicted label
      # doesn't match with the actual label.
      if predicted_label != actual_label:
        yt = self.training_labels[i]
        xt = self.training_data[i]
        self.weights_vec = self.weights_vec + yt * xt
        self.b = self.b + yt
        updates_count += 1
    return None

  def classify(self, x):
    """
        Classifies a single data point using the current perceptron model.

        Args:
            x (numpy.ndarray): A feature vector representing the data point.

        Returns:
            int: The predicted label (+1 or -1).
    """
    return np.sign(np.dot(self.weights_vec, x) + self.b)

  @property
  def error_rate(self):
    """
        Calculates the error rate on the training data.

        Returns:
            float: The proportion of misclassified examples in the training data.
    """
    misclassified = 0
    for i in range(len(self.training_data)):
      if self.classify(self.training_data[i]) != self.training_labels[i]:
        misclassified += 1
    return misclassified / len(self.training_data)
