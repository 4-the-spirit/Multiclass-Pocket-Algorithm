import itertools
import numpy as np

POSITIVE_CLASS_VALUE = 1
NEGATIVE_CLASS_VALUE = -1

BIAS_LEARNING_RATE = 5
WEIGHTS_LEARNING_RATE = 2


class PerceptronAlgorithm:
  """
    A class implementing the basic Perceptron algorithm for binary classification.

    Attributes:
        training_data (ndarray): A 2D array where each row represents a feature vector for a training example.
        training_labels (ndarray): A 1D array where each element represents the label (e.g., -1 or 1) of the corresponding training example.
        weight_vector (ndarray): A 1D array representing the weight vector of the Perceptron model.
        b (float): The bias term of the Perceptron model.

    Methods:
        (To be added in the implementation of the complete algorithm, such as training and prediction.)
  """
  def __init__(self, training_data, training_labels):
    self.training_data = training_data
    self.training_labels = training_labels
    self.weight_vector = np.zeros(len(self.training_data[0]))
    self.b = 0

  def update_parameters(self, weight_vector, b):
    self.weight_vector = weight_vector
    self.b = b
    return self

  def run(self, updates, max_epochs=1):
    updates_count = 0
    epochs_count = 0

    for i in itertools.cycle(range(len(self.training_data))):
      # Check if we arrived the number of iterations or the max epochs.
      if updates_count >= updates or epochs_count >= max_epochs:
        break

      if i == len(self.training_data) - 1:
        epochs_count += 1

      actual_label = self.training_labels[i]
      predicted_label = self.classify_single(self.training_data[i])

      # Update the weights only if the predicted label
      # doesn't match with the actual label.
      if predicted_label != actual_label:
        yt = self.training_labels[i]
        xt = self.training_data[i]
        self.weight_vector = self.weight_vector + WEIGHTS_LEARNING_RATE * yt * xt
        self.b = self.b + BIAS_LEARNING_RATE * yt
        updates_count += 1
    return None

  def classify_single(self, data):
    return np.sign(np.dot(data, self.weight_vector.T) + self.b)

  def classify(self):
    return np.sign(np.dot(self.training_data, self.weight_vector.T) + self.b)

  @property
  def error_rate(self):
    predictions = self.classify()
    misclassified = predictions != self.training_labels
    return np.sum(misclassified) / len(self.training_labels)
