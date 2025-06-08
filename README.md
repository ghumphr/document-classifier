# document-classifier
A document classifier that uses a network of CNNs, almost 100% vibecoded with Gemini

```
$ python3 dc.py 
usage: dc.py [-h] {train-hierarchical,predict-hierarchical} ...

Command-line utility for hierarchical document categorization using TensorFlow.

positional arguments:
  {train-hierarchical,predict-hierarchical}
                        Available commands
    train-hierarchical  Train hierarchical document categorization models.
    predict-hierarchical
                        Predict categories for new documents using hierarchical models.

options:
  -h, --help            show this help message and exit
```
