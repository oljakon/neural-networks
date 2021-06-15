from datasets import training_dataset, test_dataset, NUMBER_COUNT
import random


class Neuron:
    def __init__(self, f_initialize):
        self.input_weights = []
        self.bias = 0
        self.initialize = f_initialize

    def init_weights(self, count):
        for _ in range(count):
            self.input_weights.append(self.initialize())
        self.bias = self.initialize()

    def reinit_weights(self):
        self.input_weights = [self.initialize() for _ in self.input_weights]
        self.bias = self.initialize()

    def solve(self, inputs):
        raise NotImplementedError

    def correct(self, expected_result):
        pass


class ActivationNeuron(Neuron):
    def __init__(self, f_initialize, f_activate):
        super().__init__(f_initialize)
        self.last_inputs = None
        self.last_result = None
        self.activate = f_activate

    def accumulate(self, inputs):
        accumulation = - self.bias
        for value, weight in zip(inputs, self.input_weights):
            accumulation += value * weight
        return accumulation

    def solve(self, inputs):
        self.last_inputs = inputs
        self.last_result = self.activate(self.accumulate(inputs))
        return self.last_result


class NeuronS(Neuron):
    def __init__(self, f_initialize, f_transform):
        super().__init__(f_initialize)
        self.transform = f_transform

    def solve(self, inputs):
        return self.transform(inputs)


class NeuronA(ActivationNeuron):
    def calculate_bias(self):
        self.bias = 0
        for weight in self.input_weights:
            if weight > 0:
                self.bias += 1
            if weight < 0:
                self.bias -= 1


class NeuronR(ActivationNeuron):
    def __init__(self, f_initialize, f_activate, learning_speed, bias):
        super().__init__(f_initialize, f_activate)
        self.learning_speed = learning_speed
        self.bias = bias

    def correct(self, expected_result):
        if expected_result != self.last_result:
            self.input_weights = [
                input_weight - self.last_result * self.learning_speed * last_input
                for input_weight, last_input in zip(self.input_weights, self.last_inputs)
            ]
            self.bias += self.last_result * self.learning_speed


class NeuronLayer:
    def __init__(self):
        self.neurons = []

    def reinit_weights(self):
        for neuron in self.neurons:
            neuron.reinit_weights()

    def solve(self, inputs):
        raise NotImplementedError

    def correct(self, expected_results):
        pass


class LayerS(NeuronLayer):
    def add_neuron(self, f_initialize, f_transform):
        neuron = NeuronS(f_initialize, f_transform)
        self.neurons.append(neuron)

    def solve(self, inputs):
        results = []
        for neuron, value in zip(self.neurons, inputs):
            results.append(neuron.solve(value))
        return results


class LayerA(NeuronLayer):
    def add_neuron(self, inputs_count, f_initialize, f_activate):
        neuron = NeuronA(f_initialize, f_activate)
        neuron.init_weights(inputs_count)
        self.neurons.append(neuron)

    def solve(self, inputs):
        results = []
        for neuron in self.neurons:
            results.append(neuron.solve(inputs))
        return results


class LayerR(NeuronLayer):
    def add_neuron(self, inputs_count, f_initialize, f_activate, learning_speed, bias):
        neuron = NeuronR(f_initialize, f_activate, learning_speed, bias)
        neuron.init_weights(inputs_count)
        self.neurons.append(neuron)

    def solve(self, inputs):
        results = []
        for neuron in self.neurons:
            results.append(neuron.solve(inputs))
        return results

    def correct(self, expected_results):
        for neuron, expected_result in zip(self.neurons, expected_results):
            neuron.correct(expected_result)


class Perceptron:
    def __init__(self):
        self.s_layer = LayerS()
        self.a_layer = LayerA()
        self.r_layer = LayerR()

    def solve(self, inputs):
        s_result = self.s_layer.solve(inputs)
        a_result = self.a_layer.solve(s_result)
        return self.r_layer.solve(a_result)

    def correct(self, expected_results):
        self.r_layer.correct(expected_results)

    def train(self, dataset):
        continue_training = True
        epoch = 0

        total_classifications = len(dataset) * len(dataset[0].results)
        min_wrong_classifications = total_classifications
        stability_time = 0
        while continue_training and stability_time < 100:
            wrong_classifications = 0
            continue_training = False
            for data in dataset:
                results = self.solve(data.inputs)

                for result, expected_result in zip(results, data.results):
                    if result != expected_result:
                        wrong_classifications += 1
                        self.correct(data.results)
                        continue_training = True

            epoch += 1

            if min_wrong_classifications <= wrong_classifications:
                stability_time += 1
            else:
                min_wrong_classifications = wrong_classifications
                stability_time = 0

        print(f'{epoch} epochs trained\n')

    def optimize(self, dataset):
        activations = []
        for _ in self.a_layer.neurons:
            activations.append([])
        a_inputs = [self.s_layer.solve(data.inputs) for data in dataset]
        for i_count, a_input in enumerate(a_inputs):
            for n_count, neuron in enumerate(self.a_layer.neurons):
                activations[n_count].append(neuron.solve(a_input))
        to_remove = [False] * len(self.a_layer.neurons)

        a_layer_size = len(self.a_layer.neurons)
        for i, activation in enumerate(activations):
            zeros = activation.count(0)
            if zeros == 0 or zeros == a_layer_size:
                to_remove[i] = True
        dead_neurons = to_remove.count(True)
        for i in range(len(activations) - 1):
            if not to_remove[i]:
                for j in range(i + 1, len(activations)):
                    if activations[j] == activations[i]:
                        to_remove[j] = True
        correlating_neurons = to_remove.count(True) - dead_neurons

        print(f'Dead neurons: {dead_neurons}\nCorrelating neurons: {correlating_neurons}')

        for i in range(len(to_remove) - 1, -1, -1):
            if to_remove[i]:
                del self.a_layer.neurons[i]
                for j in range(len(self.r_layer.neurons)):
                    del self.r_layer.neurons[j].input_weights[i]

    def train_perceptron(self):
        input_count = len(training_dataset[0].inputs)
        for _ in range(input_count):
            self.s_layer.add_neuron(None, lambda value: value)

        a_neurons_count = 2 * input_count - 1
        for position in range(a_neurons_count):
            neuron = NeuronA(None, lambda value: int(value >= 0))
            neuron.input_weights = [
                random.choice([-1, 0, 1]) for i in range(input_count)
            ]
            neuron.calculate_bias()
            self.a_layer.neurons.append(neuron)

        for _ in range(NUMBER_COUNT):
            self.r_layer.add_neuron(a_neurons_count, lambda: 0, lambda value: 1 if value >= 0 else -1, 0.01, 0)

        self.train(training_dataset)
        self.optimize(training_dataset)

    def predict(self):
        total_classifications = len(test_dataset)
        right_answers = 0
        for data in test_dataset:
            results = self.solve(data.inputs)
            if results == data.results:
                right_answers += 1

        print('Prediction accuracy: {:.2f}%'.format(right_answers / total_classifications * 100))


def main():
    perceptron = Perceptron()
    perceptron.train_perceptron()
    perceptron.predict()


if __name__ == '__main__':
    main()
