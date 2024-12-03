from pocket_algorithm import PocketAlgorithm


class MultiClassPocketAlgorithm:
  """
    Implementation of a Multi-Class Classifier using the Pocket Algorithm.

    This algorithm extends the binary classification Pocket Algorithm to handle 
    multi-class classification problems by training a separate classifier for each class
    in a one-vs-all manner. The best weights and bias for each classifier are determined
    using the Pocket Algorithm.
  """
  
  def __init__(self, training_data, regular_training_labels):
    self._training_data = training_data
    self._regular_training_labels = regular_training_labels

    self._classification_classes = None
    self._weights_vec = None
    self._b = None

  @property
  def classification_classes(self):
    if self._classification_classes is None:
      self._classification_classes = list(set(tuple(x.astype(int)) for x in self.regular_training_labels))
    return self._classification_classes

  @property
  def weights_vec(self):
    if self._weights_vec is None:
      self._weights_vec = np.zeros((len(self.classification_classes), len(self.training_data[0])))
    return self._weights_vec

  @weights_vec.setter
  def weights_vec(self, value):
    self.weights_vec = value

  @property
  def b(self):
    if self._b is None:
      self._b = np.zeros(len(self.classification_classes))
    return self._b

  @b.setter
  def b(self, value):
    self._b = value

  @property
  def training_data(self):
    return self._training_data

  @property
  def regular_training_labels(self):
    return self._regular_training_labels

  @property
  def misclassified(self):
    misclassified = 0
    for i in range(len(self.training_data)):
      if self.classify(self.training_data[i]) != tuple(self.regular_training_labels[i].astype(int).tolist()):
        misclassified += 1
    return misclassified

  @property
  def error_rate(self):
    return self.misclassified / len(self.regular_training_labels)

  def apply_mapping(self, mapping):
    return [mapping[tuple(label.tolist())] for label in self.regular_training_labels]

  def create_mapping(self, classification_class):
    mapping = dict()
    for label in self.classification_classes:
      if label == classification_class:
        mapping[label] = PerceptronAlgorithm.POSITIVE_CLASS_VALUE
      else:
        mapping[label] = PerceptronAlgorithm.NEGATIVE_CLASS_VALUE
    return mapping

  def run(self, updates, max_epochs=1):
    for index in range(len(self.classification_classes)):
      initial_weights_vec = self.weights_vec[index]
      initial_b = self.b[index]
      positive_label = self.classification_classes[index]
      mapping = self.create_mapping(positive_label)
      training_labels = self.apply_mapping(mapping)

      pocket = PocketAlgorithm(self.training_data, training_labels).from_parameters(initial_weights_vec, initial_b)
      pocket.run(updates, max_epochs)
      self.weights_vec[index] = pocket.weights_vec
      self.b[index] = pocket.b
    return None

  def argmax(self, data):
    return np.argmax([(np.dot(self.weights_vec[i], data) + self.b[i]) for i in range(len(self.weights_vec))])

  def classify(self, data):
    return self.classification_classes[self.argmax(data)]
