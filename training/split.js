// Creates data/split.json: a fixed test set, so that every model is measured
// on exactly the same images.
//
// The dataset has some identical images (the same drawing submitted more than
// once). All copies of an image go to the same side of the split, otherwise
// the test set would contain images the network has already seen.
const { readJson, writeJson, mulberry32, shuffle } = require('./common');

const SEED = 2019;
const TEST_SHARE = 0.1;

const data = readJson('dataset.json');
const groups = new Map();
data.forEach((sample, index) => {
  const key = sample.image.join(',');
  if (!groups.has(key)) groups.set(key, []);
  groups.get(key).push(index);
});

const groupList = shuffle([...groups.values()], mulberry32(SEED));
const test = [];
for (const group of groupList) {
  if (test.length >= data.length * TEST_SHARE) break;
  test.push(...group);
}
test.sort((a, b) => a - b);

writeJson('split.json', {
  description: 'Indices of data/dataset.json in the fixed test set. Created by training/split.js.',
  seed: SEED,
  test,
});
console.log(`${data.length} samples, ${groups.size} unique images, ${test.length} in the test set`);
