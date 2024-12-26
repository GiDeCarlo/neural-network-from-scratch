import numpy as np
import pandas as pd

def load_mnist(path, shuffle=True):
  data = pd.read_csv(path)
  data = np.array(data)
  m, n = data.shape

  if shuffle: np.random.shuffle(data)

  data_dev = data[0:1000].T
  Y_dev = data_dev[0]
  X_dev = data_dev[1:n]
  X_dev = X_dev / 255.

  data_train = data[1000:m].T
  Y_train = data_train[0]
  X_train = data_train[1:n]
  X_train = X_train / 255.

  return X_dev, Y_dev, X_train, Y_train