---   
<div align="center">
 
# Ward2ICU


[![Paper](http://img.shields.io/badge/paper-arxiv.1910.00752-B31B1B.svg)](https://arxiv.org/abs/1910.00752)
[![3778 Research](http://img.shields.io/badge/3778-Research-4b44ce.svg)](https://research.3778.care/projects/privacy/)
[![3778 Research](http://img.shields.io/badge/3778-Survey-4b44ce.svg)](https://forms.gle/e2asYSVaiuPUUCKu8)

</div>

<!--ts-->
   * [Ward2ICU](#ward2icu)
      * [Description](#description)
      * [Models](#models)
         * [1D Conditional CNN GAN](#1d-conditional-cnn-gan)
         * [Recursive GAN (RGAN)](#recursive-gan-rgan)
         * [Recursive Conditional GAN (RCGAN)](#recursive-conditional-gan-rcgan)
         * [RNN Classifier](#rnn-classifier)
         * [1D-CNN Classifier](#1d-cnn-classifier)
      * [Citation](#citation)

<!-- Added by: severo, at: Tue Oct 22 03:59:12 -03 2019 -->

<!--te-->
 
## Description
Ward2ICU: A Vital Signs Dataset of Inpatients from the General Ward

## Models

### 1D Conditional CNN GAN
[![Source code](https://img.shields.io/badge/code-PyTorch-009900.svg)](https://github.com/3778/data-synthesis/blob/master/ward2icu/models/cnngan.py)

![Table 3](assets/table3.png)

### Recursive GAN (RGAN)
[![Source code](https://img.shields.io/badge/code-PyTorch-009900.svg)](https://github.com/3778/data-synthesis/blob/master/ward2icu/models/rgan.py)
[![Paper](http://img.shields.io/badge/paper-arxiv.1706.02633-B31B1B.svg)](https://arxiv.org/abs/1706.02633)

Recursive GAN (Generator) implementation with RNN cells.

### Recursive Conditional GAN (RCGAN)
[![Source code](https://img.shields.io/badge/code-PyTorch-009900.svg)](https://github.com/3778/data-synthesis/blob/master/ward2icu/models/rcgan.py)
[![Paper](http://img.shields.io/badge/paper-arxiv.1706.02633-B31B1B.svg)](https://arxiv.org/abs/1706.02633)

Recursive Conditional GAN (Generator) implementation with RNN cells

### RNN Classifier
[![Source code](https://img.shields.io/badge/code-PyTorch-009900.svg)](https://github.com/3778/data-synthesis/blob/master/ward2icu/models/classifiers.py)

A simple RNN for classification tasks. It consists of a recurrent layer (Elman RNN, LSTM or GRU) followed by 2 fully connected. The first shares parameters across the time domain (i.e. second tensor dimension), while the second collapses the time-domain to a single point with a Sigmoid activation.

### 1D-CNN Classifier
[![Source code](https://img.shields.io/badge/code-PyTorch-009900.svg)](https://github.com/3778/data-synthesis/blob/master/ward2icu/models/classifiers.py)

Single-dimension convolutional network for classification. Consists of a sequence of `Conv1d` followed by `MaxPool1d` and `Linear` with a `Sigmoid` output.

## Citation   
```
@article{severo2019ward2icu,
  title={Ward2ICU: A Vital Signs Dataset of Inpatients from the General Ward},
  author={Severo, Daniel and Amaro, Fl{\'a}vio and Hruschka Jr, Estevam R and Costa, Andr{\'e} Soares de Moura},
  journal={arXiv preprint arXiv:1910.00752},
  year={2019}
}
```
