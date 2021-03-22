# tf2-generalized-factorization-machine

This package implements a tensorflow 2.x version of the generalized factorization machine. In the case of this package, generalized means that the responses can come from different distributions, e.g. Poission, zero-inflated Gaussian, etc.

This is an early version of the package, which has only basic functionality. Features which will come in the near future:

* More distributions via [tf2-dist-utils](https://github.com/RoetGer/tf2-dist-utils)
* More versions of factorization machines via [tf2-fm-zoo](https://github.com/RoetGer/tf2-fm-zoo)

### Simple Example

```python
import tensorflow as tf
import numpy as np
import pandas as pd

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.datasets import load_boston

from tf2_dist_utils.distributions import TransNormal
from tf2_gfm import GenFactMachine


X, y = load_boston(return_X_y=True)

X = X[:,:3]
y = tf.cast(y, dtype=tf.float32)

# Current version of the package works only with discrete input
kbd = KBinsDiscretizer(n_bins=15, encode="ordinal")

nunique_vals = pd.DataFrame(X).nunique()
X = tf.cast(kbd.fit_transform(X), dtype=tf.int64)

gfm = GenFactMachine(
    target_dist=TransNormal,
    feature_cards=tf.cast(nunique_vals, tf.int32), 
    factor_dim=3)

gfm.fit(X, y)

pd.DataFrame(gfm.hist.history).plot(figsize=(15,10))
```
