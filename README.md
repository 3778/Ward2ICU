---   
<div align="center">
 
# Ward2ICU


[![Paper](http://img.shields.io/badge/paper-arxiv.1910.00752-B31B1B.svg)](https://arxiv.org/abs/1910.00752)
[![3778 Research](http://img.shields.io/badge/3778-Research-4b44ce.svg)](https://research.3778.care/projects/privacy/)
[![3778 Research](http://img.shields.io/badge/3778-Survey-4b44ce.svg)](https://research.3778.care/projects/survey)

</div>
 
## Description
Ward2ICU: A Vital Signs Dataset of Inpatients from the General Ward

## Models

### RNN Classifier
[![Source code](https://img.shields.io/badge/source%20code-009900.svg)](https://github.com/3778/data-synthesis/blob/master/ward2icu/models/classifiers.py)

A simple RNN for classification tasks. It consists of a recurrent layer (Elman RNN, LSTM or GRU) followed by 2 fully connected. The first shares parameters across the time domain (i.e. second tensor dimension), while the second collapses the time-domain to a single point with a Sigmoid activation.

### CNN-1D Classifier
[![Source code](https://img.shields.io/badge/source%20code-009900.svg)](https://github.com/3778/data-synthesis/blob/master/ward2icu/models/classifiers.py)

Single-dimension convolutional network for classification. Consists of a sequence of `Conv1d` followed by `MaxPool1d` and `Linear` with a `Sigmoid` output. An example for `kernel_size=3`, `n_layers=3` and `step_up=2` is shown below.

### Citation   
```
@article{severo2019ward2icu,
  title={Ward2ICU: A Vital Signs Dataset of Inpatients from the General Ward},
  author={Severo, Daniel and Amaro, Fl{\'a}vio and Hruschka Jr, Estevam R and Costa, Andr{\'e} Soares de Moura},
  journal={arXiv preprint arXiv:1910.00752},
  year={2019}
}
```
