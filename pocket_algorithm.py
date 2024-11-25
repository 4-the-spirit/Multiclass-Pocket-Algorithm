from perceptron_algorithm import PerceptronAlgorithm


class PocketAlgorithm(PerceptronAlgorithm):
  def __init__(self, training_data, training_labels):
    super().__init__(training_data, training_labels)

  def run(self, iterations):
    perceptron = PerceptronAlgorithm(self.training_data, self.training_labels)
    iterations_count = 0

    for i in itertools.cycle(range(len(self.training_data))):
      if iterations_count >= iterations:
        break
      old_weights_vec = perceptron.weights_vec.copy()
      old_b = perceptron.b

      perceptron.run(iterations=1)

      new_weights_vec = perceptron.weights_vec
      new_b = perceptron.b

      old_error_rate = perceptron.from_parameters(old_weights_vec, old_b).error_rate()
      new_error_rate = perceptron.from_parameters(new_weights_vec, new_b).error_rate()

      if new_error_rate < old_error_rate:
        perceptron = perceptron.from_parameters(new_weights_vec, new_b)
      else:
        perceptron = perceptron.from_parameters(old_weights_vec, old_b)

      iterations_count += 1

    self.weights_vec = perceptron.weights_vec
    self.b = perceptron.b
    return None