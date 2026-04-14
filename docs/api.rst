#############
API Reference
#############

Getting started is easy.

Unet
~~~~
.. autofunction:: segmodels_keras.Unet

Linknet
~~~~~~~
.. autofunction:: segmodels_keras.Linknet

FPN
~~~
.. autofunction:: segmodels_keras.FPN

PSPNet
~~~~~~
.. autofunction:: segmodels_keras.PSPNet

metrics
~~~~~~~
.. autofunction:: segmodels_keras.metrics.IOUScore
.. autofunction:: segmodels_keras.metrics.FScore

losses
~~~~~~
.. autofunction:: segmodels_keras.losses.JaccardLoss
.. autofunction:: segmodels_keras.losses.DiceLoss
.. autofunction:: segmodels_keras.losses.BinaryCELoss
.. autofunction:: segmodels_keras.losses.CategoricalCELoss
.. autofunction:: segmodels_keras.losses.BinaryFocalLoss
.. autofunction:: segmodels_keras.losses.CategoricalFocalLoss

utils
~~~~~
.. autofunction:: segmodels_keras.utils.set_trainable
.. autofunction:: segmodels_keras.utils.save_model_weights_notop