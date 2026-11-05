// Measures the saved plain JavaScript networks on the fixed test set.
//
//   node training/evaluate.js
//
// Note: data/NN2.json was trained in 2019 on a random 90 % of the dataset, so
// it has probably seen most of the images in today's test set. Its result here
// is therefore too optimistic. For a fair comparison, train-mlp.js 2019 trains
// the original code again on the fixed split.
const fs = require('fs');
const path = require('path');
const common = require('./common');

const lib = common.loadBrowserLibs(['lib/matrix.js', 'lib/helper.js', 'lib/neuralnetwork.js', 'lib/neuralnetwork2026.js']);
const samples = common.loadNormalizedDataset(lib);
const testOriginal = common.splitSamples(samples).test;
const hasClean = fs.existsSync(path.join(common.DATA, 'removed.json'));
const testClean = hasClean ? common.splitSamples(samples, { clean: true }).test : null;

function report(name, predict) {
  let line = `${name}: original test set ${common.accuracy(predict, testOriginal).toFixed(2)} %`;
  if (testClean) line += `, cleaned test set ${common.accuracy(predict, testClean).toFixed(2)} %`;
  console.log(line);
}

const saved2019 = common.readJson('NN2.json');
const nn2019 = new lib.NeuralNetwork(784, 300, 60, 10);
for (const key of ['weights_ih1', 'weights_h1h2', 'weights_h2o', 'bias_h1', 'bias_h2', 'bias_o']) {
  nn2019[key].values = saved2019[key].values;
}
report('NN2.json (2019)', (image) => nn2019.feedforward(image));

if (fs.existsSync(path.join(common.DATA, 'NN2026.json'))) {
  const nn2026 = lib.NeuralNetwork2026.fromJSON(common.readJson('NN2026.json'));
  report('NN2026.json', (image) => nn2026.feedforward(image));
}
