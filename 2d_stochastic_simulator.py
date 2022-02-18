# -*- coding: utf-8 -*-
"""2d stochastic simulator.ipynb
"""

from tensorflow.keras import layers
import sys
import tensorflow as tf
import numpy as np
from tensorflow.random import uniform
from tensorflow.math import sin,cos,log,sqrt,pow,exp,abs,square,multiply,reduce_sum
import math as m
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Activation, Concatenate, GaussianNoise, Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from tensorflow.keras import Input
from tensorflow.linalg import trace as tr
from tensorflow.linalg import inv,matmul
from matplotlib import pyplot as plt
import time
from tensorflow.keras import regularizers
from sklearn.neighbors import KernelDensity as kd
import scipy as scipy

"""
Data Generation
"""

seed=123
tf.random.set_seed(seed)

# These variables will later be introduced into the gaussian kernel
lambda1 = tf.Variable(0.0,dtype = tf.float32, trainable=True)
beta1 = tf.Variable(2.0,dtype=tf.float32 ,trainable=True)

pi = tf.constant(m.pi)
z3d = tfp.distributions.Normal(loc=0.,scale=1.)

#Training Data
n = 60
r = 50
a = tf.zeros([1],tf.float32)
x1_train = tf.Variable(a)
x2_train = tf.Variable(a)
y_train = tf.Variable(a)
z_train = tf.Variable(a)
for i in range(n):
  x2 = uniform([1],minval=0.1,maxval=0.4,dtype=tf.float32)
  for j in range(r):
    x1 = uniform([1],minval=0.,maxval=0.1,dtype=tf.float32)
    z1 = z3d.sample((1,))
    y = exp((x1- ((x2**2)/2)) + (z1*x2))
    y_train = tf.concat([y_train,y],axis=-1)
    x1_train = tf.concat([x1_train,x1],axis=-1)
    x2_train = tf.concat([x2_train,x2],axis=-1)

x1_train = tf.slice(x1_train,[1],[n*r])
x2_train = tf.slice(x2_train,[1],[n*r])
y_train = tf.slice(y_train,[1],[n*r])
plt.figure(figsize=(12,8))
ax = plt.axes(projection ="3d")
# Creating plot
ax.scatter3D(x1_train, x2_train, y_train, color = "green")
plt.title("scatter plot")
plt.show()

n_val = 30
r_val = 20
a = tf.zeros([1],tf.float32)
x1_val = tf.Variable(a)
x2_val = tf.Variable(a)
y_val = tf.Variable(a)

for i in range(n):
  x2 = uniform([1],minval=0.1,maxval=0.4,dtype=tf.float32)
  for j in range(r):
    x1 = uniform([1],minval=0.,maxval=0.1,dtype=tf.float32)
    z1 = z3d.sample((1,))
    y = exp((x1- ((x2**2)/2)) + (z1*x2))
    y_val = tf.concat([y_val,y],axis=-1)
    x1_val = tf.concat([x1_val,x1],axis=-1)
    x2_val = tf.concat([x2_val,x2],axis=-1)

x1_val = tf.slice(x1_val,[1],[n_val*r_val])
x2_val = tf.slice(x2_val,[1],[n_val*r_val])
y_val = tf.slice(y_val,[1],[n_val*r_val])

#Dataset Creation
batch_size = 300
train_dataset = tf.data.Dataset.from_tensor_slices((x1_train, x2_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=900).batch(batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices((x1_val, x2_val, y_val))
val_dataset = val_dataset.shuffle(buffer_size=600).batch(batch_size)

"""Network Definition and Compilation"""

n_v_r =18 # Number of Random Normal Variables
input_size = n_v_r + 2
inputs = Input(shape=(input_size,))
l1 = Dense(500, activation='relu')(inputs)
l2 = Dense(400,activation='relu')(l1)
l3 = Dense(400,activation='relu')(l2)
l4 = Dense(200,activation='relu')(l3)
outputs = Dense(1, activation='linear')(l4)
model = keras.Model(inputs=inputs , outputs=outputs, name="CGMMN_model")

model.summary()
plt.figure()
plot_model(model, to_file='cgmmn.png',show_shapes=True)
plt.show()

"""Loss Calculation
"""

### Gram Matrix
def gram_matrix(x,y, gamma1, pi1):
  sq_dist = tf.reduce_sum((tf.expand_dims(x, 1)-tf.expand_dims(y, 0))**2,2)
  kernel = (pi1*tf.exp(-gamma1*sq_dist)) 
  return kernel

### CMMD Based Loss
def multidot(A,B,C,D):
  return tf.tensordot(A,tf.tensordot(B,tf.tensordot(C,D,axes=1),axes=1),axes=1)

def cgmmn_loss(input,y_true,y_pred, lambda1, beta1):
  a = input.shape
  a= a[0]
  y_true = tf.reshape(y_true,[a,1])
  y_pred = tf.reshape(y_pred,[a,1]) 
  gamma1 = tf.math.sigmoid(lambda1)
  
  #Mixture Coefficients   
  pi1 = tf.math.exp(beta1)
  #Gram Matrices
  k_d = gram_matrix(input, input, gamma1, pi1)
  k_s = k_d 
  l_d = gram_matrix(y_true, y_true, gamma1, pi1)
  l_s = gram_matrix(y_pred, y_pred, gamma1, pi1)
  k_sd = k_d
  l_ds = gram_matrix(y_true,y_pred, gamma1, pi1)
  lam = 9.0
  row,col = k_d.shape
  k_d_tilda = k_d + (lam*tf.eye(row,col))
  k_s_tilda = k_d_tilda
  k_sd_tilda = k_d_tilda
  loss = sqrt(tr(multidot(k_d,inv(k_d_tilda),l_d,inv(k_d_tilda))) + tr(multidot(k_s,inv(k_s_tilda),l_s,inv(k_s_tilda))) - 2*tr(multidot(k_sd,inv(k_d_tilda),l_ds,inv(k_s_tilda))))
  return loss

"""Custom Training and Validation Functions"""

@tf.function
def train_step(x1,x2,z1,y, lambda1, beta1):
  a = x1.shape[0]
  with tf.GradientTape(persistent= True) as tape:
    xtemp = tf.concat([tf.reshape(x1,[a,1]),tf.reshape(x2,[a,1]),z1],axis=1)
    xtrain = tf.stack([x1,x2],axis=1)
    logits = model(xtemp, training=True)
    loss_value = cgmmn_loss(xtrain, y, logits, lambda1, beta1)
    grad1 = tape.gradient(loss_value, [lambda1, beta1])
    grads = tape.gradient(loss_value, model.trainable_weights)
    opt.apply_gradients(zip(grad1, [lambda1, beta1]))
    opt.apply_gradients(zip(grads, model.trainable_weights))
  return loss_value

@tf.function
def val_step(x1,x2,z1,y, lambda1, beta1):
  a = x1.shape[0]
  xtemp = tf.concat([tf.reshape(x1,[a,1]),tf.reshape(x2,[a,1]),z1],axis=1)
  xval = tf.stack([x1,x2],axis=1)
  logits = model(xtemp, training=False)
  loss_value = cgmmn_loss(xval, y, logits, lambda1, beta1)
  return loss_value

def testfunction(x1_value, x2_value, n= 1500):
  x1 = tf.ones([n,1])
  x1 = x1*x1_value
  x2 = tf.ones([n,1])
  x2 = x2*x2_value
  z1 = z3d.sample((n,n_v_r))
  xtemp = tf.concat([x1,x2,z1],axis=1)
  y_pred = model.predict(xtemp) 
  return y_pred

"""Optimizer Definition"""

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=10000,
    decay_rate=0.9)
#opt = keras.optimizers.Adam(learning_rate=lr_schedule)
opt = keras.optimizers.Adam(learning_rate=0.000095)

"""Training"""

!mkdir -p saved_model

# Commented out IPython magic to ensure Python compatibility.
epochs =400
train_loss_results = []
val_loss = []
minimum = 4
i = 0
initial_time = time.time()
for epoch in range(epochs):
  epoch_time = time.time()
  if (epoch % 2 == 0):
     z_samp = z3d.sample((batch_size ,n_v_r),seed=172)
  print("\nStart of epoch %d" % (i,))
  epoch_loss_avg = tf.keras.metrics.Mean()
  for step, (x1_batch_train,x2_batch_train, y_batch_train) in enumerate(train_dataset):
    loss_value = train_step(x1_batch_train,x2_batch_train,z_samp, y_batch_train, lambda1, beta1)
    epoch_loss_avg.update_state(loss_value)
    if step % 30 == 0:
          print(
                "Training loss (for one batch) at step %d: %.4f"
#                  % (step, float(loss_value))
              )
          print("Seen so far: %s samples" % ((step + 1) * batch_size))
  train_loss_results.append(epoch_loss_avg.result())

  #Validation Loss
  val_loss_avg = tf.keras.metrics.Mean()
  for step, (x1_batch_val,x2_batch_val, y_batch_val) in enumerate(val_dataset):
    val_loss_value = val_step(x1_batch_val,x2_batch_val, z_samp, y_batch_val, lambda1, beta1)
    val_loss_avg.update_state(val_loss_value)
  val_loss.append(val_loss_avg.result())
  if val_loss[i] < minimum:
      model.save('saved_model/min_error')
      minimum = val_loss[i]
  print("The train loss per Epoch is %.4f"%(epoch_loss_avg.result()))
  print("\nThe val loss per Epoch is %.4f" %(val_loss_avg.result()))
  i = i+1
  print("Per Epoch time taken: %.2fs" % (time.time() - epoch_time))
print("Total time taken: %.2fs" % (time.time() - initial_time))

"""Hellinger distance computation"""

from sklearn.model_selection import GridSearchCV
def kde_gaussian(y_d, y_preds,mini,maxi,num): 
  bandwidths1 = np.linspace(mini, maxi, num)
  grid1 = GridSearchCV(kd(kernel = 'gaussian'),{'bandwidth': bandwidths1})
  grid1.fit(y_preds);
  bd1 = grid1.best_estimator_.bandwidth
  kde = kd(bandwidth=bd1, kernel='gaussian')
  kde.fit(y_preds)
  logprob = kde.score_samples(y_d[:,None])
  value_pred = tf.constant(np.exp(logprob), dtype=tf.float32)
  return value_pred,bd1

def kde_gaussian_act(y_d,y_act,mini,maxi,num):   
  bandwidths1 = np.linspace(mini, maxi, num)
  grid2 = GridSearchCV(kd(kernel = 'gaussian'),{'bandwidth': bandwidths1})
  grid2.fit(y_act);
  bd2 = grid2.best_estimator_.bandwidth
  kde1 = kd(bandwidth=bd2, kernel='gaussian')
  kde1.fit(y_act)
  logprob1 = kde1.score_samples(y_d[:,None])
  value_actual = tf.constant(np.exp(logprob1), dtype=tf.float32)
  return value_actual,bd2

x1_value = 0.08
x2_value = 0.375
n_samp = 2000

#Predicted Data
y_pred = testfunction(x1_value, x2_value, n_samp)

#data_generation
x1 = x1_value
x2 = x2_value
yactual1 = exp((x1- ((x2**2)/2)) + (z3d.sample((n_samp,))*x2))

yactual1 = yactual1.numpy().reshape([n_samp,1])
y_d1 = np.linspace(0.,3.5,2000)
value_pred1,bd11 = kde_gaussian(y_d1, y_pred,mini = 0,maxi = 1,num = 50)
value_act1,bd12 = kde_gaussian_act(y_d1,yactual1,mini = 0,maxi = 1,num = 50)

def hellinger(p, q):
    return norm(np.sqrt(p) - np.sqrt(q)) /np.sqrt(2)

from scipy.linalg import norm
h1 = hellinger(value_pred1,value_act1) * np.sqrt(np.abs(y_d1[1]-y_d1[0]))
print(f"hellinger distance: {h1}")
