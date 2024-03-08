import numpy as np
from sklearn.datasets import load_digits


def devide_into_sets(inputs, targets, train_percent, test_percent):
    data_size = np.shape(inputs)[0]

    mapped_inputs_targets = np.array(list(zip(inputs, targets)), dtype=object)

    np.random.shuffle(mapped_inputs_targets)

    shuffled_inputs, shuffled_targets = zip(*mapped_inputs_targets) # происходит unzip

    train_size = int(data_size * train_percent)
    test_size = int(data_size * test_percent)

    train_inputs = np.array(shuffled_inputs[0 : train_size])
    train_targets = np.array(shuffled_targets[0 : train_size])

    test_inputs = np.array(shuffled_inputs[train_size : (train_size + test_size)])
    test_targets = np.array(shuffled_targets[train_size : (train_size + test_size)])

    validation_inputs = np.array(shuffled_inputs[(train_size + test_size) : ])
    validation_targets = np.array(shuffled_targets[(train_size + test_size) : ])

    return train_inputs, train_targets, test_inputs, test_targets, validation_inputs, validation_targets


def standardize(inputs, train_inputs, test_inputs, validation_inputs):
    means = np.mean(inputs, axis=0, keepdims=True)
    stds = np.std(inputs, axis=0, keepdims=True)

    std_train_inputs = (train_inputs - means) / (stds + (stds == 0))
    std_test_inputs = (test_inputs - means) / (stds + (stds == 0))
    std_validation_inputs = (validation_inputs - means) / (stds + (stds == 0))

    return std_train_inputs, std_test_inputs, std_validation_inputs


def calc_one_hot_encoding(targets):
    size = np.size(targets)
    k = np.max(targets) + 1
    one_hot = np.zeros((size, k))
    one_hot[np.arange(size), targets] = 1
    return one_hot


class LogisticRegression():
    def __init__(self, inputs, targets, reg_coeff):
        self.d = len(inputs[0])
        self.k = np.max(targets) + 1
        self.rc = reg_coeff
        self.gamma = 0.01
        self.bias = np.random.normal(0, np.sqrt(2 / (self.d + self.k)), size=(1, self.k))
        self.weights = np.random.normal(0, np.sqrt(2 / (self.d + self.k)), size=(self.k, self.d))


    def train(self, train_inputs, train_targets, epoch_cnt):
        for _ in range(epoch_cnt):
            model_confidence = self._calc_model_confidence(train_inputs)
            self.bias, self.weights = self._calc_new_weights(model_confidence, train_inputs, train_targets)

            predictions = np.argmax(model_confidence, axis=0)
            target_function_value = self._calc_target_function_value(train_inputs, train_targets)
            accuracy_on_train = self._calc_accuracy(predictions, np.argmax(train_targets, axis=1))
            confusion_matrix_on_train = self._calc_confusion_matrix(predictions, np.argmax(train_targets, axis=1))

            print(target_function_value)
            print(accuracy_on_train)
            print(confusion_matrix_on_train)


    def _calc_model_confidence(self, train_inputs):
        z = self._calc_model_output(train_inputs)
        y = self._calc_softmax(z)
        return y


    def _calc_new_weights(self, model_confidence, train_inputs, train_targets):
        bias = self.bias - self.gamma * self._calc_grad_b(model_confidence, train_targets)
        weights = self.weights - self.gamma * self._calc_grad_w(model_confidence, train_inputs, train_targets)
        return bias, weights


    def _calc_target_function_value(self, train_inputs, train_targets, z=None):
        if z is None:
            z = self._calc_model_output(train_inputs)
        
        matrix = np.tile(np.log(np.sum(np.exp(z), axis=0)), (self.k, 1))
        target_function_value = np.sum(np.diagonal(train_targets @ (matrix - z)))

        reg_matrix = self.rc / (2 * np.sum(np.power(self.weights, 2)))

        return target_function_value - reg_matrix


    def _calc_accuracy(self, predictions, targets):
        correct_predictions = np.sum(predictions == targets)
        all_predictions = predictions.shape[0]
        return correct_predictions / all_predictions


    def _calc_confusion_matrix(self, predictions, targets):
        unique_classes = np.unique(np.concatenate((predictions, targets)))

        num_classes = len(unique_classes)
        result = np.zeros((num_classes, num_classes), dtype=int)

        for i in range(num_classes):
            for j in range(num_classes):
                result[i, j] = np.sum((targets == unique_classes[i]) & (predictions == unique_classes[j]))

        return result


    def _calc_model_output(self, train_inputs):
        return self.weights @ train_inputs.T + self.bias.T


    def _calc_softmax(self, model_output):
        stability_model_output = model_output - np.max(model_output, axis=0)
        exp_model_output = np.exp(stability_model_output)
        return exp_model_output / np.sum(exp_model_output, axis=0)


    def _calc_grad_b(self, model_confidence, train_targets):
        result = (model_confidence.T - train_targets).T @ np.ones(model_confidence.T.shape[0])
        return result


    def _calc_grad_w(self, model_confidence, train_inputs, train_targets):
        result = (model_confidence.T - train_targets).T @ train_inputs + self.rc * self.weights
        return result


digits = load_digits()
inputs = digits.data
targets = digits.target

encoded_targets = calc_one_hot_encoding(targets)

tr_inputs, tr_targets, test_inputs, test_targets, val_inputs, val_targets \
    = devide_into_sets(inputs, encoded_targets, train_percent=0.75, test_percent=0.15)

std_tr_inputs, std_test_inputs, std_val_inputs = standardize(inputs, tr_inputs, test_inputs, val_inputs)

model = LogisticRegression(inputs, targets, reg_coeff=0.1)

model.train(tr_inputs, tr_targets, epoch_cnt=10)