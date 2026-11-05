// Shared helpers for the Node.js training scripts (2026 update).
// The browser files in lib/ are loaded as they are, so the training scripts
// use exactly the same code as the demo.
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const ROOT = path.join(__dirname, '..');
const DATA = path.join(ROOT, 'data');

function loadBrowserLibs(files) {
  const context = vm.createContext({ console });
  for (const file of files) {
    const code = fs.readFileSync(path.join(ROOT, file), 'utf8');
    vm.runInContext(code, context, { filename: file });
  }
  // Class declarations are not properties of the context, so expose them.
  vm.runInContext(
    'this.Matrix = typeof Matrix !== "undefined" ? Matrix : undefined;' +
      'this.NeuralNetwork = typeof NeuralNetwork !== "undefined" ? NeuralNetwork : undefined;' +
      'this.NeuralNetwork2026 = typeof NeuralNetwork2026 !== "undefined" ? NeuralNetwork2026 : undefined;',
    context
  );
  return context;
}

// Makes Math.random inside the loaded browser code repeatable.
function seedLibs(context, seed) {
  context.__seededRandom = mulberry32(seed);
  vm.runInContext('Math.random = __seededRandom;', context);
}

// Small seeded random number generator, so that every run is repeatable.
function mulberry32(seed) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffle(array, random) {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
  return array;
}

function readJson(file) {
  return JSON.parse(fs.readFileSync(path.join(DATA, file), 'utf8'));
}

function writeJson(file, value, pretty = false) {
  fs.writeFileSync(path.join(DATA, file), JSON.stringify(value, null, pretty ? 2 : 0) + '\n');
}

// The dataset with every image run through the same normalize() as the demo.
function loadNormalizedDataset(lib) {
  const raw = readJson('dataset.json');
  return raw.map((sample, index) => ({
    index,
    label: sample.label,
    image: lib.normalize(sample.image.slice()),
  }));
}

// Fixed split, see split.js. Returns { train, test } as arrays of samples.
function splitSamples(samples, { clean = false } = {}) {
  const split = readJson('split.json');
  const removed = clean ? new Set(readJson('removed.json').map((r) => r.index)) : new Set();
  const testSet = new Set(split.test);
  const keep = (s) => !removed.has(s.index);
  return {
    train: samples.filter((s) => !testSet.has(s.index) && keep(s)),
    test: samples.filter((s) => testSet.has(s.index) && keep(s)),
  };
}

function argmax(values) {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}

function accuracy(predict, samples) {
  let correct = 0;
  for (const s of samples) if (argmax(predict(s.image)) === s.label) correct++;
  return (100 * correct) / samples.length;
}

module.exports = {
  ROOT,
  DATA,
  loadBrowserLibs,
  seedLibs,
  mulberry32,
  shuffle,
  readJson,
  writeJson,
  loadNormalizedDataset,
  splitSamples,
  argmax,
  accuracy,
};
