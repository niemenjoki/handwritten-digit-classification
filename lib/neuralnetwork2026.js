// The same network as neuralnetwork.js (784-300-60-10, sigmoid hidden layers,
// softmax output, one sample at a time), with the bugs found in 2026 fixed.
// The original file is kept unchanged so that the 2019 network still works.
//
// What changed and why:
// 1. Weights start small (Xavier initialisation). With random weights between
//    -1 and 1 and 784 inputs, the hidden neurons saturate at 0 or 1 and learn
//    very slowly.
// 2. Inputs are inverted: ink is 1 and the white background is 0. Before,
//    almost every input was 1 and the drawing itself was the "missing" part.
// 3. The output gradient is (target - output). That is the correct gradient
//    for softmax with cross-entropy loss. The old code multiplied it with the
//    derivative of the sigmoid, which the output layer does not use.
// 4. The errors of the hidden layers are calculated with the weights as they
//    were in the forward pass, before they are updated, and from the
//    gradient instead of the raw error.
// 5. Softmax subtracts the largest value first, so Math.exp cannot overflow.
class NeuralNetwork2026 {
  constructor(inputs, hidden1, hidden2, outputs) {
    this.inputs = inputs;
    this.hidden1 = hidden1;
    this.hidden2 = hidden2;
    this.outputs = outputs;
    this.weights_ih1 = new Matrix(this.hidden1, this.inputs);
    this.weights_h1h2 = new Matrix(this.hidden2, this.hidden1);
    this.weights_h2o = new Matrix(this.outputs, this.hidden2);
    this.bias_h1 = new Matrix(this.hidden1, 1);
    this.bias_h2 = new Matrix(this.hidden2, 1);
    this.bias_o = new Matrix(this.outputs, 1);
    xavier(this.weights_ih1);
    xavier(this.weights_h1h2);
    xavier(this.weights_h2o);
    this.learning_rate = 0.05;
    this.learned = 0;
  }

  setLearningRate(lr) {
    this.learning_rate = lr;
  }

  static fromJSON(data) {
    const nn = new NeuralNetwork2026(data.inputs, data.hidden1, data.hidden2, data.outputs);
    for (const key of ['weights_ih1', 'weights_h1h2', 'weights_h2o', 'bias_h1', 'bias_h2', 'bias_o']) {
      nn[key].values = data[key].values;
    }
    nn.learning_rate = data.learning_rate;
    nn.learned = data.learned;
    return nn;
  }

  forward(input_array) {
    const inputs = Matrix.fromArray(input_array);
    inputs.map(mapInputs2026);

    const hidden1 = Matrix.multiply(this.weights_ih1, inputs);
    hidden1.add(this.bias_h1);
    hidden1.map(sigmoid2026);

    const hidden2 = Matrix.multiply(this.weights_h1h2, hidden1);
    hidden2.add(this.bias_h2);
    hidden2.map(sigmoid2026);

    const outputs = Matrix.multiply(this.weights_h2o, hidden2);
    outputs.add(this.bias_o);
    stableSoftmax(outputs);
    return { inputs, hidden1, hidden2, outputs };
  }

  feedforward(input_array) {
    return this.forward(input_array).outputs.toArray();
  }

  train(input_array, label) {
    this.learned++;
    const { inputs, hidden1, hidden2, outputs } = this.forward(input_array);
    const targets = new Matrix(this.outputs, 1);
    targets.values[label][0] = 1;

    // Output layer: softmax + cross-entropy gives simply (target - output).
    const output_gradient = Matrix.subtract(targets, outputs);

    // Hidden layer 2, using the weights before they are updated.
    const hidden2_errors = Matrix.multiply(Matrix.transpose(this.weights_h2o), output_gradient);
    const hidden2_gradient = Matrix.map(hidden2, dsigmoid2026);
    hidden2_gradient.multiply(hidden2_errors);

    // Hidden layer 1.
    const hidden1_errors = Matrix.multiply(Matrix.transpose(this.weights_h1h2), hidden2_gradient);
    const hidden1_gradient = Matrix.map(hidden1, dsigmoid2026);
    hidden1_gradient.multiply(hidden1_errors);

    // Update all weights and biases.
    output_gradient.multiply(this.learning_rate);
    this.weights_h2o.add(Matrix.multiply(output_gradient, Matrix.transpose(hidden2)));
    this.bias_o.add(output_gradient);

    hidden2_gradient.multiply(this.learning_rate);
    this.weights_h1h2.add(Matrix.multiply(hidden2_gradient, Matrix.transpose(hidden1)));
    this.bias_h2.add(hidden2_gradient);

    hidden1_gradient.multiply(this.learning_rate);
    this.weights_ih1.add(Matrix.multiply(hidden1_gradient, Matrix.transpose(inputs)));
    this.bias_h1.add(hidden1_gradient);
  }
}

function xavier(matrix) {
  const limit = Math.sqrt(6 / (matrix.rows + matrix.cols));
  matrix.map(() => (Math.random() * 2 - 1) * limit);
}

function stableSoftmax(matrix) {
  const max = Math.max(...matrix.toArray());
  matrix.map((x) => Math.exp(x - max));
  const sum = matrix.toArray().reduce((a, b) => a + b, 0);
  matrix.map((x) => x / sum);
}

function sigmoid2026(x) {
  return 1 / (1 + Math.exp(-x));
}

function dsigmoid2026(y) {
  // y = sigmoid(x)
  return y * (1 - y);
}

function mapInputs2026(x) {
  // 255 (white background) -> 0, 0 (black ink) -> 1
  return (255 - x) / 255;
}
