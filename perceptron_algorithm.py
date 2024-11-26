import itertools
import numpy as np


class PerceptronAlgorithm:
  """
  The labels of the training data must be either 1 or -1.
  """
  POSITIVE_CLASS_VALUE = 1
  NEGATIVE_CLASS_VALUE = -1

  def __init__(self, training_data, training_labels):
    self.training_data = training_data
    self.training_labels = training_labels
    self.weights_vec = np.zeros(len(self.training_data[0]))
    self.b = 0

  def from_parameters(self, weights_vec, b):
    perceptron = PerceptronAlgorithm(self.training_data, self.training_labels)
    perceptron.weights_vec = weights_vec
    perceptron.b = b
    return perceptron

  def run(self, updates, max_epochs=1):
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
        print(f"Perceptron Algorithm: {round((updates_count / updates) * 100, 2)} %")
    return None

  def classify(self, x):
    return np.sign(np.dot(self.weights_vec, x) + self.b)

  @property
  def error_rate(self):
    misclassified = 0
    for i in range(len(self.training_data)):
      if self.classify(self.training_data[i]) != self.training_labels[i]:
        misclassified += 1
    return misclassified / len(self.training_data)
