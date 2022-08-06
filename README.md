# DSLR (Data Science X Logistic Regression)

This project is about making *Logistic Regression* using *Gradient Descent* on a (Harry Potter themed) fake dataset.

## Data Visualization

`histogram.py` answers the question: \
"Which Hogwarts course has a homogeneous score distribution between all four houses?"

`scatter_plot.py` answers the question: \
"What are the two features that are similar ?"

`pair_plot.py` answers the question: \
"What features are you going to use for your logistic regression?"

## Usage

```
$ python logreg_train.py --help
usage: logreg_train.py [-h] [--dataset DATASET] [--alpha ALPHA]
                       [--max_iter MAX_ITER] [--show] [--accuracy] [--loss]
                       [-r]

Train model with logistic regression

optional arguments:
  -h, --help           show this help message and exit
  --dataset DATASET    dataset used (default: ./datasets/train.csv)
  --alpha ALPHA        define learning rate (default: 0.1)
  --max_iter MAX_ITER  define number of iterations (default: 7000)
  --show               display plots during gradient descent
  --accuracy           plot the accuracy
  --loss               plot the loss
  -r                   plot the data repartition
```
