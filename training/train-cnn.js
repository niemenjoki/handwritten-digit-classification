// Trains a small convolutional network with TensorFlow.js as a point of
// comparison for the plain JavaScript network. Needs @tensorflow/tfjs-node
// (see training/package.json).
//
//   node training/train-cnn.js                    train on the fixed split, report test accuracy
//   node training/train-cnn.js --clean            the same on the cleaned dataset
//   node training/train-cnn.js --clean --save     ... and save the model to data/tfjs-model/
//   node training/train-cnn.js --oof --out=x.json 5-fold predictions for every image,
//                                                 used to find suspicious labels (clean.js)
const fs = require('fs');
const path = require('path');
const tf = require('@tensorflow/tfjs-node');
const common = require('./common');

const opt = Object.fromEntries(
  process.argv
    .slice(2)
    .filter((a) => a.startsWith('--'))
    .map((a) => {
      const [k, v] = a.slice(2).split('=');
      return [k, v === undefined ? true : v];
    })
);
const EPOCHS = Number(opt.epochs || 15);
const SEED = Number(opt.seed || 1);

function createModel(seed) {
  const init = () => tf.initializers.glorotUniform({ seed: seed++ });
  const model = tf.sequential();
  model.add(tf.layers.conv2d({ inputShape: [28, 28, 1], filters: 32, kernelSize: 3, activation: 'relu', kernelInitializer: init() }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu', kernelInitializer: init() }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dropout({ rate: 0.4, seed: seed++ }));
  model.add(tf.layers.dense({ units: 128, activation: 'relu', kernelInitializer: init() }));
  model.add(tf.layers.dropout({ rate: 0.4, seed: seed++ }));
  model.add(tf.layers.dense({ units: 10, activation: 'softmax', kernelInitializer: init() }));
  model.compile({ optimizer: tf.train.adam(0.001), loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
  return model;
}

// Same input format as in the demo: ink 1, background 0.
function toTensors(samples) {
  const xs = new Float32Array(samples.length * 784);
  samples.forEach((s, i) => s.image.forEach((v, j) => (xs[i * 784 + j] = (255 - v) / 255)));
  return {
    xs: tf.tensor4d(xs, [samples.length, 28, 28, 1]),
    ys: tf.oneHot(tf.tensor1d(samples.map((s) => s.label), 'int32'), 10),
  };
}

async function train(model, samples, seed) {
  const random = common.mulberry32(seed);
  for (let epoch = 1; epoch <= EPOCHS; epoch++) {
    const { xs, ys } = toTensors(common.shuffle(samples.slice(), random));
    const h = await model.fit(xs, ys, { epochs: 1, batchSize: 64, shuffle: false, verbose: 0 });
    xs.dispose();
    ys.dispose();
    console.log(`epoch ${epoch}/${EPOCHS}  loss ${h.history.loss[0].toFixed(4)}`);
  }
}

function predict(model, samples) {
  const { xs, ys } = toTensors(samples);
  const out = model.predict(xs);
  const probs = out.arraySync();
  [xs, ys, out].forEach((t) => t.dispose());
  return probs;
}

function accuracyOf(probs, samples) {
  const correct = samples.filter((s, i) => common.argmax(probs[i]) === s.label).length;
  return Number(((100 * correct) / samples.length).toFixed(2));
}

async function main() {
  const lib = common.loadBrowserLibs(['lib/matrix.js', 'lib/helper.js']);
  const samples = common.loadNormalizedDataset(lib);
  const start = Date.now();

  if (opt.oof) {
    // Out-of-fold predictions: every image is predicted by a model that never saw it.
    // Identical images stay in the same fold.
    const keys = new Map();
    samples.forEach((s) => {
      const k = s.image.join(',');
      if (!keys.has(k)) keys.set(k, keys.size);
    });
    const groupOrder = common.shuffle([...Array(keys.size).keys()], common.mulberry32(SEED));
    const foldOfGroup = new Map(groupOrder.map((g, i) => [g, i % 5]));
    const fold = samples.map((s) => foldOfGroup.get(keys.get(s.image.join(','))));
    const result = new Array(samples.length);
    for (let f = 0; f < 5; f++) {
      const trainSet = samples.filter((s, i) => fold[i] !== f);
      const heldOut = samples.filter((s, i) => fold[i] === f);
      const model = createModel(SEED * 100 + f * 10);
      await train(model, trainSet, SEED + f);
      const probs = predict(model, heldOut);
      heldOut.forEach((s, i) => (result[s.index] = { index: s.index, label: s.label, probs: probs[i].map((p) => Number(p.toFixed(4))) }));
      console.log(`fold ${f + 1}/5 accuracy ${accuracyOf(probs, heldOut)} %`);
      model.dispose();
    }
    fs.writeFileSync(path.resolve(opt.out), JSON.stringify(result));
    return;
  }

  const { train: trainSet, test } = common.splitSamples(samples, { clean: Boolean(opt.clean) });
  const testOriginal = common.splitSamples(samples).test;
  const model = createModel(SEED * 100);
  await train(model, trainSet, SEED);
  const result = {
    settings: { model: 'tfjs-cnn', clean: Boolean(opt.clean), epochs: EPOCHS, seed: SEED },
    seconds: Math.round((Date.now() - start) / 1000),
    testAccuracy: accuracyOf(predict(model, test), test),
    testAccuracyOriginalTestSet: accuracyOf(predict(model, testOriginal), testOriginal),
    train: trainSet.length,
    test: test.length,
  };
  console.log(JSON.stringify(result));
  if (opt.out) fs.writeFileSync(path.resolve(opt.out), JSON.stringify(result));
  if (opt.save) {
    const dir = path.join(common.DATA, 'tfjs-model');
    await model.save('file://' + dir);
    console.log('saved to data/tfjs-model/');
  }
}

main();
