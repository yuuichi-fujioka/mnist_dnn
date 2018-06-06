# MNIST + DNN

This sample uses premade estimator `DNNClassifier`.

## Requirements

Python3

```
$ pip install tensorflow
$ pip install pandas
$ pip install prettytable
$ pip install plyvel
```

## How To Use

### Step1: Training

```
$ python training.py
```

`models` directory will be created when training is finished.


### Step2: Evaluate

```
$ python eval.py
```

Evaluate the model.

### Step3: Prediction

```
$ python predict.py
```

Do Prediction.

### Step4: show Result

```
$ python stat_log.py
```

## Misc

If You want to add test pattern(e.g. changing hidden_units, batch_size, train_step), you may change `test_iter` function at `util.py`.
