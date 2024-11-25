

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

  def from_parameters(self, weight_vec, b):
    self.weights_vec = weight_vec
    self.b = b
    return self

  def run(self, iterations):
    iterations_count = 0
    for i in itertools.cycle(range(len(self.training_data))):
      if iterations_count >= iterations:
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
      iterations_count += 1
    return None

  def classify(self, x):
    return np.sign(np.dot(self.weights_vec, x) + self.b)

  def error_rate(self):
    misclassified = 0
    for i in range(len(self.training_data)):
      if self.classify(self.training_data[i]) != self.training_labels[i]:
        misclassified += 1
    return misclassified / len(self.training_data)