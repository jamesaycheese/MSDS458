# Handwritten digit recognition

Assignment 1 for MSDS 458 at Northwestern: ten experiments with fully connected neural networks on the MNIST handwritten digits.

Try the best model in the browser, with a photo or a drawing of a digit: https://jamesquantai.com/projects/digit-recognition.html

## The experiments

Each network trains on 55,000 digits for 30 epochs, is checked on 5,000 validation digits, and is scored on the 10,000 test digits.

| Experiment | Network | Test accuracy |
|---|---|---|
| 1 | 1 hidden node | 40.2% |
| 2 | 2 hidden nodes | 72.1% |
| 3 | 32, 16, 8 nodes, ReLU | 96.3% |
| 4 | 64, 32, 16 nodes, tanh | 97.2% |
| **5** | **128, 64, 32 nodes, batch normalization** | **98.1%** |
| 6 | 128, 64, 32 nodes, dropout | 97.9% |
| 7 | 128, 64, 32 nodes, AdamW | 97.7% |
| 8 | 64, 64, 64 nodes, batch normalization, SGD | 98.0% |
| 9 | 85 nodes, all 784 pixels | 97.5% |
| 9 | 85 nodes, 154 PCA components | 97.5% |
| 10 | 85 nodes, 70 pixels chosen by a random forest | 93.9% (validation) |

Accuracies are from the notebook. Experiment 10 reports validation accuracy only.

## Files

| File | What it is |
|---|---|
| `MSDS458_Assignment1.ipynb` | All ten experiments, with confusion matrices and hidden-node activation plots |
| `train_experiment5_numpy.py` | Experiment 5 retrained in plain NumPy, so its weights can be exported. Reaches 98.1% test accuracy |
| `experiment5_weights.json` | Weights, training history and test confusion matrix from one run of that script |

The notebook did not save its trained models, so the browser demo uses the weights in `experiment5_weights.json`. The code that prepares a photo for the model (grey scale, shadow removal, cropping, scaling to 28 by 28 and centring) is in the website repo: [`js/digits.js`](https://github.com/jamesaycheese/jamesaycheese.github.io/blob/main/js/digits.js).

## Running the training script

It needs only `numpy` and the four original MNIST `.gz` files in one folder:

```bash
python train_experiment5_numpy.py /path/to/mnist/
```
