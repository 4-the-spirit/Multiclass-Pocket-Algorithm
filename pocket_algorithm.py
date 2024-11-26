import itertools
import numpy as np
from perceptron_algorithm import PerceptronAlgorithm


class PocketAlgorithm(PerceptronAlgorithm):
  def __init__(self, training_data, training_labels):
    super().__init__(training_data, training_labels)

  def run(self, updates, max_epochs=1):
    """
    For each iteration in the Pocket Algorithm in which the weights vector is updated
    this method will execute until it reaches the number of updates or until the number
    of maximum epochs was reached.
    """
    perceptron = PerceptronAlgorithm(self.training_data, self.training_labels)
    updates_count = 0
    epochs_count = 0

    for i in itertools.cycle(range(len(self.training_data))):
      if i == len(self.training_data) - 1:
        epochs_count += 1
      if updates_count >= updates or epochs_count >= max_epochs:
        break

      old_weights_vec = perceptron.weights_vec.copy()
      old_b = perceptron.b

      perceptron.run(updates=1, max_epochs=1)

      new_weights_vec = perceptron.weights_vec.copy()
      new_b = perceptron.b

      old_error_rate = perceptron.from_parameters(old_weights_vec, old_b).error_rate
      new_error_rate = perceptron.from_parameters(new_weights_vec, new_b).error_rate

      if new_error_rate < old_error_rate:
        perceptron.weights_vec = new_weights_vec
        perceptron.b = new_b
        updates_count += 1
        print(f"Pocket Algorithm: {round((updates_count / updates) * 100, 2)} %")

      self.weights_vec = perceptron.weights_vec
      self.b = perceptron.b

    return None

