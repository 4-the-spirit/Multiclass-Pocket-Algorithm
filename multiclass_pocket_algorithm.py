from pocket_algorithm import PocketAlgorithm


class MultiClassPocketAlgorithm:
  """
    An implementation of the Multi-Class Pocket Algorithm for classification tasks
    with multiple classes. This algorithm trains a separate binary Pocket Algorithm
    model for each class in a one-vs-all approach.

    Attributes:
        training_data (ndarray): A 2D array where each row represents a feature vector for a training example.
        regular_training_labels (ndarray): A 2D array where each row represents the one-hot encoded label
                                           for the corresponding training example.
        classification_classes (list): A list of unique class labels, where each label is represented as a tuple.
        weight_vectors (ndarray): A 2D array of shape (n_classes, n_features) where each row contains the weight
                                  vector corresponding to a particular class.
        b (ndarray): A 1D array of shape (n_classes,) containing the bias term for each class.
        _errors (list): A list storing the error rates observed for each class during training.

    Methods:
        (To be implemented, such as running the multi-class Pocket Algorithm or accessing error rates.)
  """

  def __init__(self, training_data, regular_training_labels):
    self.training_data = training_data
    self.regular_training_labels = regular_training_labels

    self.classification_classes = list(set(tuple(x.astype(int)) for x in self.regular_training_labels))

    self.weight_vectors = np.zeros((len(self.classification_classes), len(self.training_data[0])))
    self.b = np.zeros(len(self.classification_classes))

  @property
  def error_rate(self):
    misclassified = 0
    for i in range(len(self.training_data)):
      if self.classify_single(self.training_data[i]) != tuple(self.regular_training_labels[i].astype(int).tolist()):
        misclassified += 1
    return misclassified / len(self.training_data)

  @property
  def errors(self):
    return np.array(self._errors)

  def apply_mapping(self, mapping):
    return [mapping[tuple(label.tolist())] for label in self.regular_training_labels]

  def create_mapping(self, classification_class):
    mapping = dict()
    for label in self.classification_classes:
      if label == classification_class:
        mapping[label] = POSITIVE_CLASS_VALUE
      else:
        mapping[label] = NEGATIVE_CLASS_VALUE
    return mapping

  def run(self, updates, max_epochs=1):
    for index in range(len(self.classification_classes)):
      # Use the weights and bias that belongs to the current model.
      initial_weights_vec = self.weight_vectors[index]
      initial_b = self.b[index]

      positive_label = self.classification_classes[index]
      mapping = self.create_mapping(positive_label)
      training_labels = self.apply_mapping(mapping)

      pocket = PocketAlgorithm(self.training_data, training_labels).update_parameters(initial_weights_vec, initial_b)
      pocket.run(updates, max_epochs)

      self.weight_vectors[index] = pocket.weight_vector
      self.b[index] = pocket.b
    return None

  def argmax(self, data):
    return np.argmax([(np.dot(self.weight_vectors[i], data) + self.b[i]) for i in range(len(self.weight_vectors))])

  def classify_single(self, data):
    return self.classification_classes[self.argmax(data)]

  @property
  def confusion_matrix(self):
    mat = np.zeros((len(self.classification_classes), len(self.classification_classes)), dtype="int")

    for i in range(len(self.training_data)):
      predicted_class = self.classify_single(self.training_data[i])
      predicted_class_index = self.classification_classes.index(predicted_class)
      actual_class = tuple(self.regular_training_labels[i].astype(int).tolist())
      actual_class_index = self.classification_classes.index(actual_class)
      mat[actual_class_index][predicted_class_index] += 1

    return mat
