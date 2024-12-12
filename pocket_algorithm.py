import itertools
import numpy as np
from perceptron_algorithm import PerceptronAlgorithm


class PocketAlgorithm(PerceptronAlgorithm):
  """
    An implementation of the Pocket Algorithm, extending the PerceptronAlgorithm.
    The Pocket Algorithm is designed for binary classification, maintaining the
    best weights and bias observed during training to handle non-linearly separable data.

    Attributes:
        training_data (ndarray): A 2D array where each row represents a feature vector for a training example.
        training_labels (ndarray): A 1D array where each element represents the label (e.g., -1 or 1) of the corresponding training example.
        weight_vector (ndarray): A 1D array representing the current weight vector of the model.
        b (float): The current bias term of the model.
        _errors (list): A list storing the error rates observed at different stages of training.

    Inherits:
        PerceptronAlgorithm: The base Perceptron algorithm, providing basic functionality for binary classification.

    Methods:
        (To be implemented in the full algorithm, such as running the Pocket Algorithm or accessing error rates.)
  """
  def __init__(self, training_data, training_labels):
    super().__init__(training_data, training_labels)

  def run(self, updates, max_epochs=1):
    updates_count = 0
    epochs_count = 0

    # Store initial error rate and weights as the best ones
    self.best_error_rate = self.error_rate
    self.best_weight_vector = self.weight_vector.copy()
    self.best_b = self.b

    for i in itertools.cycle(range(len(self.training_data))):
        if updates_count >= updates or epochs_count >= max_epochs:
            break

        # If we reach the end of the data, increment epoch count
        if i == len(self.training_data) - 1:
            epochs_count += 1

        # If the current sample is classified correctly, continue
        if self.classify_single(self.training_data[i]) == self.training_labels[i]:
            continue

        # Run one update of the Perceptron
        super().run(updates=1, max_epochs=1)

        # After the update, calculate the current error rate
        current_error_rate = self.error_rate

        # If the new error rate is better (lower), update the pocket parameters
        if current_error_rate < self.best_error_rate:
            self.best_weight_vector = self.weight_vector.copy()
            self.best_b = self.b
            self.best_error_rate = current_error_rate

        updates_count += 1

    # After finishing the updates, set the best weight vector and bias
    self.update_parameters(self.best_weight_vector, self.best_b)
    return None
