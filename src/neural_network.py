import numpy as np

def init_params(n_layers, input_dim, hidden_units, output_dim):
  params = {}
  for i in range(n_layers):
    params[i] = {}
    if i == 0:
      # input layer
      params[i]['W'] = np.random.randn(hidden_units, input_dim) * np.sqrt(2. / input_dim)
      params[i]['b'] = np.zeros((hidden_units, 1))
    elif i == n_layers-1:
      # output layer
      params[i]['W'] = np.random.randn(output_dim, hidden_units) * np.sqrt(2. / output_dim)
      params[i]['b'] = np.zeros((output_dim, 1))
    else:
      # hidden layers 
      params[i]['W'] = np.random.randn(hidden_units, hidden_units) * np.sqrt(2. / 10)
      params[i]['b'] = np.zeros((hidden_units, 1))
  
  return params

# ADAM State Initialisation 
def init_adam(params):
  adam = {}
  for layer in params:
    adam[layer] = {}
    for key in params[layer]:
      adam[layer][key] = {
        'm': np.zeros_like(params[layer][key]),
        'v': np.zeros_like(params[layer][key])
      }
  return adam

def ReLU(Z):
  return np.maximum(Z, 0)

def softmax(Z):
  Z_shift = Z - np.max(Z, axis=0, keepdims=True)
  exp_Z = np.exp(Z_shift)
  A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
  return A
    
def forward_prop(params, n_layers, X, keep_prob, training=True):
  Z_i = []
  A_i = [X]
  D_i = []

  for layer in range(n_layers):
    W = params[layer]['W']
    b = params[layer]['b']
    Z = W.dot(A_i[-1]) + b
    Z_i.append(Z)
    if layer == n_layers - 1:
      A = softmax(Z)
    else:
      A = ReLU(Z)
      if training:
        D = (np.random.rand(*A.shape) < keep_prob).astype(float)
        A *= D
        A /= keep_prob
        D_i.append(D)
      else:
        D_i.append(None)

    A_i.append(A)
  
  return Z_i, A_i, D_i

def ReLU_deriv(Z):
  return (Z > 0).astype(float)

def one_hot(Y):
  one_hot_Y = np.zeros((Y.size, Y.max() + 1))
  one_hot_Y[np.arange(Y.size), Y] = 1
  one_hot_Y = one_hot_Y.T
  return one_hot_Y

def backward_prop(params, Z_i, A_i, D_i, X, Y, n_layers, keep_prob):
  m = X.shape[1]
  one_hot_Y = one_hot(Y)

  grads = {}
  dA = A_i[-1] - one_hot_Y

  for layer in reversed(range(n_layers)):
    Z = Z_i[layer]
    A_prev = A_i[layer]

    if layer == n_layers - 1:
      dZ = dA
    else:
      D = D_i[layer]
      dZ = params[layer + 1]['W'].T.dot(dZ_prev) * ReLU_deriv(Z)
      if D is not None:
        dZ *= D
        dZ /= keep_prob
    
    dW = (1 / m) * dZ.dot(A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

    grads[layer] = {'dW': dW, 'db': db}
    dZ_prev = dZ

  return grads

def update_params(params, grads, alpha, n_layers):
  for layer in range(n_layers):
    params[layer]['W'] -= alpha * grads[layer]['dW']
    params[layer]['b'] -= alpha * grads[layer]['db']
  return params

def update_params_adam(params, grads, adam, t, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
  for layer in params:
    for key in params[layer]:
      if key == 'W':
        grad = grads[layer]['dW']
      elif key == 'b':
        grad = grads[layer]['db']
      else:
        raise ValueError("Unknown Key")

      # Moments update
      adam[layer][key]['m'] = beta1 * adam[layer][key]['m'] + (1 - beta1) * grad
      adam[layer][key]['v'] = beta2 * adam[layer][key]['v'] + (1 - beta2) * (grad ** 2)

      # Bias correction
      m_hat = adam[layer][key]['m'] / (1 - beta1 ** t)
      v_hat = adam[layer][key]['v'] / (1 - beta2 ** t)

      # Params update
      params[layer][key] -= alpha * m_hat / (np.sqrt(v_hat) + epsilon)
    return params, adam

def get_predictions(A2):
  return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
  return np.sum(predictions == Y) / Y.size

def compute_loss(A_final, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    log_probs = -np.log(A_final + 1e-8)
    loss = np.sum(one_hot_Y * log_probs) / m
    return loss

def train(X, Y, hidden_units, n_classes, alpha, epochs, n_layers, keep_prob):
  input_dim = X.shape[0]
  params    = init_params(n_layers, input_dim, hidden_units, n_classes)
  adam      = init_adam(params)

  best_params = None
  best_acc    = 0
  t           = 0

  train_accuracies  = []
  train_losses      = []

  for i in range(epochs):
    t += 1

    Z_i, A_i, D_i = forward_prop(params, n_layers, X, keep_prob)
    grads         = backward_prop(params, Z_i, A_i, D_i, X, Y, n_layers, keep_prob)
    params, adam  = update_params_adam(params, grads, adam, t, alpha)

    preds     = get_predictions(A_i[n_layers])
    train_acc = get_accuracy(preds, Y)
    train_lss = compute_loss(A_i[-1], Y)

    train_accuracies.append(train_acc)
    train_losses.append(train_lss)

    if train_acc > best_acc:
      best_acc = train_acc
      best_params = params
      
    if i % 10 == 0 or i == epochs-1:
      print(f"Epoch {i}: Train Accuracy = {train_acc:.3f} |  Train Loss = {train_lss:.3f} | Best Accuracy = {best_acc:.3f}")
      
  return best_params, train_accuracies, train_losses