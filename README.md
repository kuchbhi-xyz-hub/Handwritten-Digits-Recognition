# Handwritten-Digits-Recognition
MNIST digit recognition: EDA, data checks, and a comparison of LR, KNN, SVM, MLP and CNN. CNN picked for production (~99% test accuracy).
# Handwritten Digits Recognition (PRCP-1002)

Classifying handwritten digit images (0–9) from the MNIST dataset and comparing classical ML and deep learning models to find the most suitable one for production.

The whole project lives in a single Jupyter notebook: data analysis, preprocessing, five models, error analysis, model comparison, a production recommendation, and a challenges report.

---

## Problem Statement

The project had three tasks:

1. Prepare a complete data analysis report on the handwritten digit data.
2. Classify a handwritten digit image into one of 10 classes (0–9).
3. Compare multiple models and recommend the best one for production.

It also asked for a report on the challenges faced and the techniques used to handle them.

## Dataset

The supplied project folder didn't contain any images. It had a Word document instructing that the data be loaded with:

```python
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

What I found after inspecting it:

- 60,000 training and 10,000 test images
- Each image is 28×28 grayscale, with pixel values from 0 to 255
- 10 classes (digits 0–9), with no missing values
- Mild class imbalance (max/min ratio of 1.24, with digit 1 the most frequent and digit 5 the least)
- About 81% of pixels are pure background, so the images are sparse

## Approach

1. **Data validation:** I checked shapes, dtypes, pixel range, labels and missing values, plus blank images, duplicates and train/test overlap.
2. **EDA:** I looked at the class distribution, sample images per digit, pixel intensity distribution, and the mean image per class. The mean images were used to predict which digit pairs might be confused.
3. **Preprocessing:** I scaled pixels to [0, 1], flattened images to 784 features for the classical models, and reshaped them back to 28×28×1 for the CNN.
4. **Split strategy:** I carved a stratified 10,000-image validation set out of the training data. The official test set was used only once, for the final model.
5. **Modeling:** I trained every model under one shared evaluation function so the comparison stays fair.
6. **Evaluation:** I measured accuracy plus macro precision, recall and F1, alongside confusion matrices, per-class reports, misclassified-image inspection, and training and inference time.

## Models Compared

| Model | Why it's included |
|---|---|
| Majority-class baseline | Absolute performance floor |
| Logistic Regression | Linear reference point |
| K-Nearest Neighbors | Named in the project's practice skills; k chosen on a subset |
| SVM (RBF kernel) | Named in the project's practice skills; training cost probed before the full run |
| MLP (dense neural network) | Simple neural network on the same flattened input |
| CNN | Tests whether keeping the 2D image structure helps |

## Results (validation set)

| Model | Accuracy | F1 (macro) | ms / image |
|---|---:|---:|---:|
| CNN | 0.9875 | 0.9874 | 0.096 |
| SVM (RBF) | 0.9826 | 0.9825 | 6.049 |
| MLP | 0.9769 | 0.9768 | 0.063 |
| KNN (k=3) | 0.9707 | 0.9707 | 0.656 |
| Logistic Regression | 0.9227 | 0.9217 | 0.004 |

Timings depend on hardware. These come from a single run on a local CPU.

**Selected model: CNN.** On the held-out test set it reached **99.01% accuracy** with a macro F1 of 0.9900, which works out to 99 errors out of 10,000 images.

## Why the CNN

The selection wasn't based on accuracy alone:

- **SVM** was less accurate than the CNN and about 63× slower per image. It also has to keep around 10,000 support vectors in memory.
- **KNN** was less accurate and about 7× slower per image, and it needs the full training set stored at prediction time.
- **The MLP** was the closest contender. It's slightly faster and trains in about a third of the time, but the CNN's accuracy gain of roughly 1 percentage point mattered more.

Among the evaluated models, the CNN gave the best balance of accuracy, per-class consistency and inference cost.

## Key Findings

- Every non-linear model beat logistic regression by more than 4 points, which suggests the digit classes aren't linearly separable in raw pixel space.
- KNN over-predicted digit 1 (precision 0.94), while SVM and the neural networks didn't show this problem.
- The confusions the EDA predicted (4/9, 3/8, 5/3) showed up in most models' top errors. KNN's most frequent error (8 predicted as 1) wasn't predicted, though.
- Only 57 of the 446 images that at least one model misclassified were missed by all four main models, so the models fail in partly different ways.

## Real Image Prediction

The notebook includes a pipeline to test the CNN on photographed handwriting. It uses grayscale conversion, contrast stretching, Otsu thresholding, polarity detection, cropping, and centering into a 28×28 frame. It was first sanity-checked on a known MNIST image, then applied to an external photo.

## Challenges

The full details are in the notebook. In short:

- **Unknown KNN/SVM runtime:** I measured cost on small subsets and extrapolated before committing to full training runs.
- **Slow SVM inference:** this was documented and weighed in the production decision.
- **Fixed thresholds failing on real photos:** I replaced them with adaptive Otsu thresholding.
- **Stroke-style sensitivity:** documented as a limitation of MNIST-trained models.
- **Run-to-run variation in neural network results:** documented. It didn't change the model ranking.

## Limitations

- Strong MNIST accuracy doesn't guarantee performance on arbitrary real-world handwriting, because the model is sensitive to stroke style and capture conditions.
- The validation set was also used for early stopping and for choosing k, so validation scores are slightly optimistic. The test set was used only once.
- Timings come from one machine and one run.

## Repository Structure

```
├── PRCP_1002_Handwritten_Digits_Recognition.ipynb   # full project notebook
├── testing-digit1.jpg                               # external image used for prediction
└── README.md
```

## How to Run

1. Install the dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn tensorflow pillow
   ```
2. Keep `testing-digit1.jpg` in the same folder as the notebook.
3. Open the notebook and run **Kernel → Restart & Run All**. The full run takes a few minutes on CPU, and the SVM is the slowest step.

MNIST downloads automatically on the first run (about 11 MB).

## Tech Stack

Python, NumPy, pandas, Matplotlib, Seaborn, scikit-learn, TensorFlow/Keras, Pillow

## Author

**Smita Sahu**
