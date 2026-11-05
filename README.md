# Handwritten digit classification

> **An old learning project from 2019, with one last update in 2026.** It is no longer developed, but the demo still works
> and the dataset is free to use.

A neural network written in plain JavaScript that recognizes digits you draw. I built it in 2019 to learn programming and
the basics of neural networks. More than 200 volunteers drew about 14 000 digits for the training data.

**[Try the demo](https://niemenjoki.github.io/handwritten-digit-classification/)**: draw a digit with your mouse or finger,
and the bars show how confident the network is about each digit. The menu under the canvas switches between the 2019
network, the same network with its bugs fixed, and a TensorFlow.js comparison model.

## 2026 update

Before archiving the project I went back to it with seven more years of experience. The goal was to keep the original
code and find out how good it could have been, not to rewrite it.

| Model | Test accuracy | Training time |
|---|---|---|
| 2019 network, original code | 89.0 % | 29 min |
| 2019 network, bugs fixed | 96.0 % | 29 min |
| 2019 network, bugs fixed, cleaned dataset | 96.9 % | 21 min |
| TensorFlow.js convolutional network, cleaned dataset | 98.8 % | 2 min |

All models were trained with the same settings as in 2019 (20 epochs) and measured on the same fixed test set of 1 395
digits. With the cleaned dataset the test set has 1 371 digits, because the faulty ones were removed from it too. On the
uncleaned test set the last two models reach 96.1 % and 98.1 %. Training ran on a 6-core desktop CPU.

**What was fixed** ([lib/neuralnetwork2026.js](lib/neuralnetwork2026.js); the original
[lib/neuralnetwork.js](lib/neuralnetwork.js) is unchanged):

1. **Weight initialisation.** Weights started at random values between -1 and 1. With 784 inputs, the hidden neurons
   were pushed to 0 or 1 right away and learned very slowly. Now they start small (Xavier initialisation).
2. **Inverted inputs.** The white background was 1 and the ink 0, so almost every input was "on" and the digit itself
   was the missing part. Now ink is 1.
3. **Output gradient.** The output uses softmax, but the gradient was multiplied by the derivative of the sigmoid. With
   softmax and cross-entropy the correct gradient is simply target minus output.
4. **Backpropagation order.** The errors of the hidden layers were calculated after the output weights had already been
   updated. Now all errors are calculated first, with the weights of the forward pass.
5. **Softmax overflow.** Softmax now subtracts the largest value before `Math.exp`.

The first fix alone made the biggest difference: after one epoch the fixed network is at 90 % and the original at 79 %.

**Fixed test set.** In 2019 the test set was drawn at random on every run, so the results could not be compared.
Now the test set is fixed ([data/split.json](data/split.json)), and identical images are always on the same side of
the split.

**Cleaned dataset.** A TensorFlow.js model was trained five times so that each image was predicted by a model that had
never seen it. I looked at the 176 images where the model confidently disagreed with the label, and removed only the
clear cases. Unclear handwriting stays in the dataset, because it is part of real handwriting. 313 images were removed
([data/removed.json](data/removed.json) lists each one with the reason):

| Reason | Images |
|---|---|
| Duplicate of another image | 190 |
| Wrong label (for example a clear 7 labelled as 1) | 82 |
| Not a recognizable digit (faces, scribbles, words) | 31 |
| Several digits in one image | 7 |
| Roman numeral | 3 |

The original [data/dataset.json](data/dataset.json) is unchanged.

**About the 77 % in the 2019 README.** The 2019 README reported 77 % after 60 epochs. The network the demo has always used
(`data/NN2.json`) was, according to its own metadata, trained for 20 epochs, and the original code trained again today
reaches 89 %. The 77 % was most likely from an earlier run.

## Dataset

The dataset has 13 942 handwritten digits drawn by volunteers in 2019.

| All | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|-----|---|---|---|---|---|---|---|---|---|---|
| **13 942** | 1284 | 1441 | 1439 | 1376 | 1431 | 1357 | 1374 | 1465 | 1328 | 1447 |

![Sample of the dataset](assets/dataset.png)

Each sample has two fields:

- `image`: 784 numbers from 0 to 255, a 28 x 28 pixel image row by row (255 = white, 0 = black)
- `label`: the digit drawn

Download [data/dataset.json](data/dataset.json) (about 40 MB), or load it with JavaScript:

```javascript
const URL =
  'https://raw.githubusercontent.com/niemenjoki/handwritten-digit-classification/master/data/dataset.json';
const data = await fetch(URL).then((res) => res.json());
// Array(13942) [{ image: [...], label: 7 }, ...]
```

The same digits are also split by label in `data/numbers/0.json` to `data/numbers/9.json`.

To use the cleaned version, leave out the images listed in [data/removed.json](data/removed.json) (indices of
`dataset.json`), or run `node training/clean.js build` to write `data/dataset-clean.json`.

The dataset is dedicated to the public domain under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/): use it
for anything, no attribution needed.

## How it works

- The network is based on Daniel Shiffman's [Toy Neural Network](https://github.com/CodingTrain/Toy-Neural-Network-JS).
- Layers: 784 inputs, two hidden layers of 300 and 60 neurons, and 10 outputs.
- Training: stochastic gradient descent with backpropagation, one sample at a time, 20 epochs of 12 547 samples. The
  learning rate started at 0.01 (0.05 in the fixed network) and was halved every 5 epochs.
- Before training and prediction, each drawing is centered, cropped and scaled to 28 x 28 pixels (`lib/helper.js`).
- The only library is [p5.js](https://p5js.org/) for drawing on the canvas. The TensorFlow.js comparison model loads
  TensorFlow.js only when you select it.

## Running it locally

The demo is static HTML and JavaScript, but it loads the trained network with `fetch`, so it needs a web server:

```bash
npx serve .
```

Then open the address it prints.

The 2026 training scripts run on Node.js 18 or newer. The plain JavaScript network needs no packages. The TensorFlow.js
scripts need `@tensorflow/tfjs-node`:

```bash
cd training && npm install && cd ..
node training/train-mlp.js 2019            # the original code
node training/train-mlp.js 2026 --clean    # the fixed network on the cleaned dataset
node training/train-cnn.js --clean         # the TensorFlow.js model
node training/evaluate.js                  # the saved networks on the fixed test set
```

The original 2019 training code is in `learning/`. It has no user interface: run it from the browser console.

## Contents

| Path | Contents |
|---|---|
| `index.html`, `index.js`, `UI.js`, `style.css` | the demo |
| `lib/` | the neural network and matrix code (`neuralnetwork2026.js` is the fixed network) |
| `learning/` | the original 2019 training code (browser console) |
| `training/` | the 2026 training, cleaning and evaluation scripts (Node.js) |
| `data/dataset.json` | the dataset |
| `data/numbers/` | the dataset split by digit |
| `data/removed.json` | the images left out of the cleaned dataset, with reasons |
| `data/split.json` | the fixed test set |
| `data/NN2.json` | the 2019 network |
| `data/NN2026.json` | the fixed network, trained on the cleaned dataset |
| `data/tfjs-model/` | the TensorFlow.js comparison model |
| `data/NN.json` | an earlier, smaller network |

## License

- Code: [MIT](LICENSE).
- Dataset: [CC0 1.0](data/LICENSE).

## Acknowledgements

Inspired by Daniel Shiffman's [neural network videos](https://github.com/CodingTrain) and 3Blue1Brown's
[Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) series.

Thank you to everyone who drew digits for the dataset:

11v, 5v, \@Olli78, A, Aapo, ak, Alanne S, Aleksei K, Aleksi, Alex, Anna, Anne, Anni B, Antti Merilehto, Anu, AR-, Arma, Arttu, asiannutija, Astennuz, Atte, Azoth, brighttux, Charlotta, closet nerd :), cool, Disa, Distrobat, eeli l, Eetu Rantanen, Elfred, Elina H., Emil, EmiliaE, Emmu, Emty, Erkka, esa, Evani, G, gambina1557, h, Hanna S, Harri.kankaanpaa, Heidi, Heikki Pahnila, hjampe, homersimpson, HouBe , hxl, irreki, J0, Jaakko, Jaakko T, Janne, Janne O, Jari O, Jari Varjus, Jarkko Moilanen, Jarno Oravainen, Jenni, Jenny, JK, Johanna , Jonas Rajanto, Joni, Jonna, Jonne, Joonas, Joonas Hämäläinen, Lunanova Oy, Jounk, JS17, JT, JTLeskinen, Jukeboksi, Jukka, Julle, Jussi Aleksi Anno, Jussi R, JussiK, Jutta, JV, Jycce, Jyrki Kostilainen, jz, Kaisa, Kari Martikainen, Kata, KATJA, Katja, Katti, Kilroyr, Kim B, Kip, Kokoomusnuori, Komu, konola, KS, Lassi Mölsä, Lassi P, Laura-Maaria Penttilä , Lauri, LawlietNick, Leens, Lenni Laukkanen, Lestaatius Oli Ristitty Ihmii, LiisaP, LordIC, M, M., Maija, Mami, Marko O, Markus, Markus Valkama, Masta Niemelä, Mauno, Mervi, mh, Micke, Miia, Mika, Mika T, Mika Turunen, mikeful, Mikko, Mikko OJ, Mikko S, Mirde, MirjaA, Mirva, MLa, Mmm, Mr.Noise, Nanite, Nerdaxic, Niklas Piispanen, Niko, niko, Nina, ninnixcecilia, nokkaeläin, Nooa, noora, Noora, Numerojäbä, Olli Jernfors, Olli-Pekka Inget, Olo, Oona, Oskar, Outi, PAH, Pantter1, PanuV, Pauliina, Pauliina Murtola, Pauliina Pietilä, Pekkis, PetriP, Petrus, Petteri, Pia Klemetti, Piia Patronen, Poc, Pomppu-Jaska, Ppp, prs, Pörni, r, Rami Hirvelä, Rekku, Rick, Riina, Riku K, Roman, S. S., Saana, sadra, Salli , Sami, Sami Hakonen, Sandezu, Sari, Sari Toivonen, Sarita, SH, Siiri, simppalammas, Sini, Sini K, Snen, Suvi, sxrtu, takku, TanjaM, Taru, Taya, Tero, Tero Paajoki, Tero Valkiala, test, Tiekki, Tiia Sarell, Tiina , tiina , Timo Halttunen, Tomi, Tomi Hägg, TomiS, Tommi Partanen, Toni, Toni Asikainen, Tonisn, Topi, Tuire, Tuomas Keskinarkaus, Tuukka, Tuula Metsä , Vaasa Hacklab, Vellu, Venla, Viittakostaja, Viljanen , Ville Luukkonen, Ville Vaajanen, Vul, Winterroad

![All digits in the dataset](assets/datasetLarge.png)
