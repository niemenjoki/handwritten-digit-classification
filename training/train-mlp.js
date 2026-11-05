// Trains the plain JavaScript network from the command line.
//
//   node training/train-mlp.js 2019          the original code, unchanged
//   node training/train-mlp.js 2026          the fixed network (lib/neuralnetwork2026.js)
//   node training/train-mlp.js 2026 --clean  the fixed network on the cleaned dataset
//
// Options can be given as --epochs=20 --lr=0.05 --seed=1 --out=file.json
// The training settings of 2019 were: learning rate 0.01, 20 epochs and the
// learning rate halved every 5 epochs (learning/index.js).
const fs = require('fs');
const path = require('path');
const common = require('./common');

const args = process.argv.slice(2);
const version = args[0];
const opt = Object.fromEntries(
  args
    .filter((a) => a.startsWith('--'))
    .map((a) => {
      const [k, v] = a.slice(2).split('=');
      return [k, v === undefined ? true : v];
    })
);
if (version !== '2019' && version !== '2026') {
  console.error('Usage: node training/train-mlp.js 2019|2026 [--clean] [--epochs=N] [--lr=X] [--seed=N] [--out=file]');
  process.exit(1);
}

const settings = {
  version,
  clean: Boolean(opt.clean),
  epochs: Number(opt.epochs || 20),
  learningRate: Number(opt.lr || (version === '2019' ? 0.01 : 0.05)),
  learningRateDivider: 2,
  modifyLearningRateEvery: 5,
  seed: Number(opt.seed || 1),
};

const libFiles = ['lib/matrix.js', 'lib/helper.js', version === '2019' ? 'lib/neuralnetwork.js' : 'lib/neuralnetwork2026.js'];
const lib = common.loadBrowserLibs(libFiles);
common.seedLibs(lib, settings.seed);

const samples = common.loadNormalizedDataset(lib);
const { train, test } = common.splitSamples(samples, { clean: settings.clean });
const testOriginal = common.splitSamples(samples).test;
console.log(JSON.stringify(settings), `train ${train.length}, test ${test.length}`);

const nn = version === '2019' ? new lib.NeuralNetwork(784, 300, 60, 10) : new lib.NeuralNetwork2026(784, 300, 60, 10);
nn.setLearningRate(settings.learningRate);
const predict = (image) => nn.feedforward(image);
const random = common.mulberry32(settings.seed + 1000);
const history = [];
const start = Date.now();

for (let epoch = 1; epoch <= settings.epochs; epoch++) {
  if (epoch > 1 && (epoch - 1) % settings.modifyLearningRateEvery === 0) {
    nn.setLearningRate(nn.learning_rate / settings.learningRateDivider);
  }
  for (const s of common.shuffle(train.slice(), random)) nn.train(s.image, s.label);
  const acc = common.accuracy(predict, test);
  history.push({ epoch, learningRate: nn.learning_rate, testAccuracy: Number(acc.toFixed(2)) });
  console.log(`epoch ${epoch}/${settings.epochs}  lr ${nn.learning_rate}  test ${acc.toFixed(2)} %  ${((Date.now() - start) / 1000).toFixed(0)} s`);
}

const result = {
  settings,
  seconds: Math.round((Date.now() - start) / 1000),
  testAccuracy: Number(common.accuracy(predict, test).toFixed(2)),
  testAccuracyOriginalTestSet: Number(common.accuracy(predict, testOriginal).toFixed(2)),
  history,
};
console.log(JSON.stringify(result));

if (opt.out) {
  const outDir = path.dirname(path.resolve(opt.out));
  fs.mkdirSync(outDir, { recursive: true });
  const saved = JSON.parse(JSON.stringify(nn));
  delete saved.batchOutputs;
  delete saved.output_errors;
  saved.trainingResult = result;
  // Six significant digits are plenty and halve the file size.
  const round = (key, value) => (typeof value === 'number' ? Number(value.toPrecision(6)) : value);
  fs.writeFileSync(path.resolve(opt.out), JSON.stringify(saved, round));
}
