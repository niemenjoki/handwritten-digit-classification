// Dataset cleaning in two steps. The original data/dataset.json is never changed.
//
// 1. node training/clean.js candidates oof.json candidates.json
//    Lists the images worth a look: duplicates, and images where a model that
//    never saw the image confidently disagrees with the label (train-cnn.js --oof).
//    A person then looks at every candidate and decides.
//
// 2. node training/clean.js build
//    Reads the decisions from data/removed.json and writes data/dataset-clean.json.
const fs = require('fs');
const path = require('path');
const common = require('./common');

const [step, ...files] = process.argv.slice(2);
const data = common.readJson('dataset.json');

function duplicates() {
  const first = new Map();
  const result = [];
  data.forEach((s, index) => {
    const key = s.image.join(',');
    if (first.has(key)) result.push({ index, reason: `duplicate of image ${first.get(key)}` });
    else first.set(key, index);
  });
  return result;
}

if (step === 'candidates') {
  const oof = JSON.parse(fs.readFileSync(path.resolve(files[0]), 'utf8'));
  const suspicious = oof
    .map((r) => {
      const predicted = common.argmax(r.probs);
      return { index: r.index, label: r.label, predicted, confidence: r.probs[predicted], labelProbability: r.probs[r.label] };
    })
    .filter((r) => r.predicted !== r.label && (r.confidence >= 0.9 || r.labelProbability < 0.02))
    .sort((a, b) => a.labelProbability - b.labelProbability);
  const result = { duplicates: duplicates(), suspicious };
  fs.writeFileSync(path.resolve(files[1]), JSON.stringify(result, null, 1));
  console.log(`${result.duplicates.length} duplicates, ${suspicious.length} suspicious labels`);
} else if (step === 'build') {
  const removed = new Set(common.readJson('removed.json').map((r) => r.index));
  const clean = data.filter((s, index) => !removed.has(index));
  common.writeJson('dataset-clean.json', clean);
  console.log(`${data.length} -> ${clean.length} images (${removed.size} removed)`);
} else {
  console.error('Usage: node training/clean.js candidates <oof.json> <out.json> | build');
  process.exit(1);
}
