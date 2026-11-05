# Handwritten digit classification

> **An old learning project from 2019.** It is no longer developed, but the demo still works and the dataset is free to use.

A neural network written in plain JavaScript that recognizes digits you draw. I built it in 2019 to learn programming and
the basics of neural networks. More than 200 volunteers drew about 14 000 digits for the training data.

**[Try the demo](https://niemenjoki.github.io/handwritten-digit-classification/)**: draw a digit with your mouse or finger,
and the bars show how confident the network is about each digit.

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

The dataset is dedicated to the public domain under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/): use it
for anything, no attribution needed.

## How it works

- The network is based on Daniel Shiffman's [Toy Neural Network](https://github.com/CodingTrain/Toy-Neural-Network-JS).
- Layers: 784 inputs, two hidden layers of 300 and 60 neurons, and 10 outputs.
- Training: stochastic gradient descent with backpropagation, 60 epochs of 12 547 samples. The learning rate started at
  0.01 and was halved every 5 epochs.
- Accuracy on a test set of 1 392 digits: **77 %**.
- The only library is [p5.js](https://p5js.org/) for drawing on the canvas.

## Running it locally

The demo is static HTML and JavaScript, but it loads the trained network with `fetch`, so it needs a web server:

```bash
npx serve .
```

Then open the address it prints. The training code is in `learning/`. It has no user interface: run it from the browser
console.

## Contents

| Path | Contents |
|---|---|
| `index.html`, `index.js`, `UI.js`, `style.css` | the demo |
| `lib/` | the neural network and matrix code |
| `learning/` | training code |
| `data/dataset.json` | the dataset |
| `data/numbers/` | the dataset split by digit |
| `data/NN2.json` | the trained network used by the demo |
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
