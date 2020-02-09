# Various TensorFlow Models:
This repository contains individual scripts for useful models. They are all built with the tensorflow keras functional api. Each script for a specific model allows one to import the model and instantiate it with an `tf.keras.Input` and a `tf.keras.Model` is returned ready for training.

The goal is to take common 2D models and write more __general implementations to accomodate 1D and 3D__. These models can be useful for signal data, voxel data, image data, etc. 

## Models:

1. `ndResNet.py`: 

A ResNet where the convolutional operators match the channel dimension of the data. All the other operations such as pooling, global ops, and padding also are dependent on the channel dim of `tf.keras.Input`.

```
# example
from ndResnet import ndResNet

# setup input
input_signal = layers.Input(shape=(300,3))
signal = np.zeros((1,300,3), dtype=np.float32)

# build the model (resnet18)
signal_resnet = ndResnet(input_signal, 
             bn_axis=2, 
             n_output=20,
             size=18,
             model_name='signal_resnet')

# inference example
signal_resnet(signal).shape
>>> TensorShape([1, 20])
```

2. `ndUNet.py` (TODO)

3. `ndInception.py` (TODO)

4. `ndMobilenet.py` (TODO)

5. `ndXception.py` (TODO)

6. `ndVgg.py` (TODO)
 
