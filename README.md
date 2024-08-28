# Text Classifier
## **Introduction**
This repository contains the implementation of SVM classifier of text classes.

## **Installation**
* Make sure you had installed the requiring libraries in your system.

* Clone repository to your local machine.
 ````text
git clone https://github.com/mbkorkusuz/textClassifierSVM.git
````
* Navigate to the project directory in the terminal.

Run the `train.py` script to train the classifier with the dataset.
 ````text
python3 train.py
````
When training is done run the `test.py` script for testing the classifier

For benchmark results, run:
````text
python3 test.py
````

## **Metrics**
| | Precision | Recall |  F1-Score | Support | 
| ------------- | ------------- | ------------- | ------------- |  ------------- |
| philosophy | 0.71 | 0.61 | 0.66 | 228
| sports | 0.85 | 0.74 | 0.79 | 235 |
| mystery | 0.82 | 0.75 | 0.78 | 240 |
| science | 0.77 | 0.57 | 0.66 | 230 |
| romance | 0.65 | 0.83 | 0.73 | 470 |
| horror | 0.61 | 0.65 | 0.63 | 228 |
| science-fiction | 0.67 | 0.65 | 0.66 | 234 |
| Accuracy |   |  | 0.71 | 1865 |
| Macro AVG | 0.73 | 0.69 | 0.70 | 1865 |
| Weighted AVG | 0.72 | 0.71 | 0.70 | 1865 |




