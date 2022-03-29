# Joint optimization of Autoencoder and Self-Supervised Classifier: Anomaly Detection of Strawberries using Hyperspectral Imaging ——— A Tensorflow Implementation

## Requirements

+ python==3.x
+ tensorflow==1.15.1
+ tflearn==0.5.0
+ numpy==1.19.5
+ scikit-learn==0.20.2

## How to run

+ You can now train the SSC_AE using default parameters using
  `python3 train.py`
+ In order to get results. you can run the following command
  `python3 test.py`

## Results
you can check result:
`result.csv`
|        | A U C | F1 score | ACC_normal | ACC_bruise | ACC_decay | ACC_contaminated |
|--------|:-----:|:--------:|:----------:|:----------:|:---------:|:----------------:|
| SSC_AE | 0.908 |   0.840  |    0.820   |    0.929   |   0.717   |       0.839      |

## License

MIT © Yisen Liu