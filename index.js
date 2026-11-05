let numberPoints = [];
let eraserOn = false;
let nn = new NeuralNetwork(784, 300, 60, 10); // the 2019 network
let nn2026; // the same network with the bugs fixed (2026 update)
let tfModel; // TensorFlow.js comparison model, loaded only when selected
let activeModel = '2026';

fetch('data/NN2026.json')
  .then(res => res.json())
  .then(data => {
    nn2026 = NeuralNetwork2026.fromJSON(data);
  });

fetch('data/NN2.json')
  .then(res => res.json())
  .then(taughtnn => {
    nn.weights_ih1.values = taughtnn.weights_ih1.values;
    nn.weights_h1h2.values = taughtnn.weights_h1h2.values;
    nn.weights_h2o.values = taughtnn.weights_h2o.values;
    nn.bias_h1.values = taughtnn.bias_h1.values;
    nn.bias_h2.values = taughtnn.bias_h2.values;
    nn.bias_o.values = taughtnn.bias_o.values;
  });

function setup() {
  const canvas = createCanvas(280, 280);
  canvas.parent('canvasDiv');
  pixelDensity(1);
}

function draw() {
  background(255);
  if (numberPoints) drawPoints();
  if (eraserOn) drawEraseArea();
}

function guess() {
  let scaledP = prepdata();
  if (scaledP) {
    const confidences = predict(scaledP);
    if (!confidences) return;
    for (let i = 0; i < confidences.length; i++) {
      const elm = document.querySelector(`#confbar${i}`);
      elm.style.height = (100 * confidences[i] + 5) + '%';
    }
  }
}

function predict(input) {
  if (activeModel === '2019') return nn.feedforward(input);
  if (activeModel === '2026') return nn2026 ? nn2026.feedforward(input) : null;
  if (activeModel === 'tfjs' && tfModel) {
    return tf.tidy(() => {
      const ink = input.map(v => (255 - v) / 255);
      return Array.from(tfModel.predict(tf.tensor4d(ink, [1, 28, 28, 1])).dataSync());
    });
  }
  return null;
}

function selectModel(value) {
  activeModel = value;
  if (value === 'tfjs' && !document.querySelector('#tfjsScript')) {
    const script = document.createElement('script');
    script.id = 'tfjsScript';
    script.src = 'https://cdnjs.cloudflare.com/ajax/libs/tensorflow/4.22.0/tf.min.js';
    script.onload = () => tf.loadLayersModel('data/tfjs-model/model.json').then(model => {
      tfModel = model;
      if (numberPoints.length > 1) guess();
    });
    document.head.appendChild(script);
  }
  if (numberPoints.length > 1) guess();
}
