# handwritten-digit-classification: A computer vision project

This project was originally created to help me learn programming and the fundamentals of neural networks.

It includes code for a web app that **uses** a handwritten digit recognition model.  
The app is **no longer live**, but the source code remains available in this repository.  

A separate data collection app once existed to gather handwritten digits from users â€” that app has been **discontinued**.  
However, the **dataset** it produced is still available for anyone to download and experiment with.

---

## Dataset

The dataset was built from user-submitted drawings collected through the original web app.  
Each submission was stored in a [Google Cloud Firestore database](https://firebase.google.com/products/firestore/), then compiled into a single JSON file.

Each sample in the dataset contains:

* `image:` a 784-element array of numbers `[0â€“255]`, representing pixel values for a 28Ã—28 image  
  (`255` = white, `0` = black)
* `label:` the numeric digit corresponding to the image

### Dataset snapshot (as of 28 November, 2019):

| All | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|-----|---|---|---|---|---|---|---|---|---|---|
| **13942** | 1284 | 1441 | 1439 | 1376 | 1431 | 1357 | 1374 | 1465 | 1328 | 1447 |

![Picture of the dataset](assets/dataset.png)

### Download the dataset

You can download the dataset directly from this repository:

ðŸ‘‰ [Download dataset.json](data/dataset.json)

Or load it dynamically with JavaScript:

```javascript
let data;
(async () => {
  const URL = 'https://raw.githubusercontent.com/jnsjknn/handwritten-digit-classification/master/data/dataset.json';
  data = await fetch(URL).then(res => res.json());
})();
// data becomes: Array(13942) [{ image: [...], label: ... }, ...]
```

Use of the dataset is completely open â€” attribution is appreciated but not required.

---

## Web App

The code for the digit recognition web app is included in this repository.  
It demonstrates how the trained neural network can be used to predict digits drawn by a user on a canvas element.

Although the app is **not hosted anywhere**, it can still be run locally by cloning the repository.

---

## Neural Network

The neural network implemented here is based on [Daniel Shiffman's Toy Neural Network](https://github.com/CodingTrain/Toy-Neural-Network-JS).  
It uses stochastic gradient descent backpropagation and simple mathematical functions (not optimized for performance).

### Training summary

* Epochs: 60  
* Samples per epoch: 12 547  
* Learning rate: started at 0.01, halved every 5 epochs  
* Test set: 1 392 digits  
* Accuracy: **77 %**

Future work (if resumed) could include converting the model to TensorFlow.js for performance and usability improvements.

---

## Project status

- Dataset still available  
- Web app code included in repo  
- Web app not currently live  
- Data collection app discontinued  

This repository remains archived as a **learning resource** and **dataset source**.

---

## Credits

This project was inspired by:

- [Daniel Shiffman](https://github.com/CodingTrain) *Neural Networks Playlist*  
- [3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) *Neural Networks Explained*