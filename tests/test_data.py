import numpy as np
import pytest
from sybric.data import TimeSeriesVitalSigns


dataset = TimeSeriesVitalSigns()

def test_TimeSeriesVitalSigns_inv_transforms():
    d = dataset
    X = d.X.astype(np.float128)
    y = d.y
    s = d.synthesis_df(X, y)
    assert np.isclose(d.inv_whiten(d.whiten(X)), X).all()
    assert np.isclose(d.inv_normalize(d.normalize(X)), X).all()
    assert np.isclose(d.inv_minmax(d.minmax(X)), X).all()
    assert np.isclose(d.inv_minmax_signals(d.minmax_signals(X)), X).all()

    columns = ['cat_vital_sign', 't', 'class']
    assert (s[columns] == d.df[columns]).all().all()
    assert np.isclose(s['value'], d.df['value']).all()

def test_TimeSeriesVitalSigns_transform_minmax():
    X = np.array([
      # patient 1 
      [[36, 90], [40, 100], [40, 100]],
      # patient 2          
      [[40, 80], [36, 90],  [36, 90]],
      # patient 3          
      [[40, 80], [36, 90],  [36, 90]]
    ])

    e = np.array([
      # patient 1 
      [[-1, 1], [1, 1], [1, 1]],
      # patient 2
      [[1, -1], [-1, -1], [-1, -1]],
      # patient 3
      [[1, -1], [-1, -1], [-1, -1]]
    ])

    assert (dataset.minmax(X, X) == e).all()
    assert X.shape == (3, 3, 2)

def test_TimeSeriesVitalSigns_transform_minmax_signals():
    X = np.array([
      # patient 1
      [[36, 90], [40, 100], [40, 100]],
      # patient 2
      [[40, 80], [36, 90],  [36, 90]],
      # patient 2
      [[40, 80], [36, 90],  [36, 90]],
      # patient 2
      [[40, 80], [36, 90],  [36, 90]],
    ])

    # signal 1 = [36, 40,  40, 36] -> [-1, 1,  1, -1]
    # signal 2 = [90, 100, 80, 90] -> [ 0, 1, -1,  0]

    e = np.array([
      # patient 1
      [[-1, 0], [1,  1], [1,  1]],
      # patient 2
      [[1, -1], [-1, 0], [-1, 0]],
      # patient 2
      [[1, -1], [-1, 0], [-1, 0]],
      # patient 2
      [[1, -1], [-1, 0], [-1, 0]]
    ])

    assert (dataset.minmax_signals(X, X) == e).all()
    assert X.shape == (4, 3, 2)
    assert e.shape == (4, 3, 2)
