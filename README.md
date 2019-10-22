# Ward2ICU
Ward2ICU: A Vital Signs Dataset of Inpatients from the General Ward

# Models

## RNN Classifier
[\[Code\]](https://github.com/3778/data-synthesis/blob/master/ward2icu/models/classifiers.py)

A simple RNN for classification tasks. It consists of a recurrent layer (Elman RNN, LSTM or GRU) followed by 2 fully connected. The first shares parameters across the time domain (i.e. second tensor dimension), while the second collapses the time-domain to a single point with a Sigmoid activation.

## CNN-1D Classifier
[\[Code\]](https://github.com/3778/data-synthesis/blob/master/ward2icu/models/classifiers.py)

Single-dimension convolutional network for classification. Consists of a sequence of `Conv1d` followed by `MaxPool1d` and `Linear` with a `Sigmoid` output. An example for `kernel_size=3`, `n_layers=3` and `step_up=2` is shown below.
