segmodels_keras
=========================

This is a fork of the
`segmentation_models <https://github.com/qubvel/segmentation_models>`__ library by
Pavel Iakubovskii, which is not maintained anymore.

This fork is updated to support Keras 3, and also contains some bug fixes, some
improvements and support for some extra backbone models.

It is not meant as a full replacement of the original library, but rather as a
solution for a library I developed and depended on segmentation_models:
[orthoseg](https://github.com/orthoseg/orthoseg). Hence, full backwards
compatibility,... or support for all features is not guaranteed.


**The main features** of this library are:

-  High level API (just two lines of code to create model for segmentation)
-  **4** models architectures for binary and multi-class image segmentation
   (including legendary **Unet**)
-  **20+** available backbones for each architecture
-  All backbones have **pre-trained** weights for faster and better
   convergence
- Helpful segmentation losses (Jaccard, Dice, Focal) and metrics (IoU, F-score)


Table of Contents
~~~~~~~~~~~~~~~~~
 - `Quick start`_
 - `Simple training pipeline`_
 - `Examples`_
 - `Models and Backbones`_
 - `Installation`_
 - `Documentation`_
 - `Change log`_
 - `Citing`_
 - `License`_
 
Quick start
~~~~~~~~~~~
Library is build to work together with Keras and TensorFlow Keras frameworks

.. code:: python

    import segmodels_keras as smk
    # Segmentation Models: using `keras` framework.

By default it tries to import ``keras``, if it is not installed, it will try to start with ``tensorflow.keras`` framework.
There are several ways to choose framework:

- Provide environment variable ``SM_FRAMEWORK=keras`` / ``SM_FRAMEWORK=tf.keras`` before import ``segmodels_keras``
- Change framework ``smk.set_framework('keras')`` /  ``smk.set_framework('tf.keras')``

You can also specify what kind of ``image_data_format`` to use, segmodels_keras works with both: ``channels_last`` and ``channels_first``.
This can be useful for further model conversion to Nvidia TensorRT format or optimizing model for cpu/gpu computations.

.. code:: python

    import keras
    # or from tensorflow import keras

    keras.backend.set_image_data_format('channels_last')
    # or keras.backend.set_image_data_format('channels_first')

Created segmentation model is just an instance of Keras Model, which can be build as easy as:

.. code:: python
    
    model = smk.Unet()
    
Depending on the task, you can change the network architecture by choosing backbones with fewer or more parameters and use pretrainded weights to initialize it:

.. code:: python

    model = smk.Unet('resnet50', encoder_weights='imagenet')

Change number of output classes in the model (choose your case):

.. code:: python
    
    # binary segmentation (this parameters are default when you call Unet('resnet50')
    model = smk.Unet('resnet50', classes=1, activation='sigmoid')
    
.. code:: python
    
    # multiclass segmentation with non overlapping class masks (your classes + background)
    model = smk.Unet('resnet50', classes=3, activation='softmax')
    
.. code:: python
    
    # multiclass segmentation with independent overlapping/non-overlapping class masks
    model = smk.Unet('resnet50', classes=3, activation='sigmoid')
    
    
Change input shape of the model:

.. code:: python
    
    # if you set input channels not equal to 3, you have to set encoder_weights=None
    # how to handle such case with encoder_weights='imagenet' described in docs
    model = smk.Unet('resnet50', input_shape=(None, None, 6), encoder_weights=None)
   
Simple training pipeline
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    import segmodels_keras as smk

    BACKBONE = 'resnet50'
    preprocess_input = smk.get_preprocessing(BACKBONE)

    # load your data
    x_train, y_train, x_val, y_val = load_data(...)

    # preprocess input
    x_train = preprocess_input(x_train)
    x_val = preprocess_input(x_val)

    # define model
    model = smk.Unet(BACKBONE, encoder_weights='imagenet')
    model.compile(
        'Adam',
        loss=smk.losses.bce_jaccard_loss,
        metrics=[smk.metrics.iou_score],
    )

    # fit model
    model.fit(
       x=x_train,
       y=y_train,
       batch_size=16,
       epochs=100,
       validation_data=(x_val, y_val),
    )

Same manipulations can be done with ``Linknet``, ``PSPNet`` and ``FPN``. For more
detailed information about models API and use cases
`Read the Docs <https://segmodels-keras.readthedocs.io/en/latest/>`__.

Models and Backbones
~~~~~~~~~~~~~~~~~~~~
**Models**

-  `Unet <https://arxiv.org/abs/1505.04597>`__
-  `FPN <http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf>`__
-  `Linknet <https://arxiv.org/abs/1707.03718>`__
-  `PSPNet <https://arxiv.org/abs/1612.01105>`__

============= ==============
Unet          Linknet
============= ==============
|unet_image|  |linknet_image|
============= ==============

============= ==============
PSPNet        FPN
============= ==============
|psp_image|   |fpn_image|
============= ==============

.. _Unet: https://github.com/orthoseg/segmodels_keras/blob/main/LICENSE
.. _Linknet: https://arxiv.org/abs/1707.03718
.. _PSPNet: https://arxiv.org/abs/1612.01105
.. _FPN: http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf

.. |unet_image| image:: https://github.com/orthoseg/segmodels_keras/blob/main/images/unet.png
.. |linknet_image| image:: https://github.com/orthoseg/segmodels_keras/blob/main/images/linknet.png
.. |psp_image| image:: https://github.com/orthoseg/segmodels_keras/blob/main/images/pspnet.png
.. |fpn_image| image:: https://github.com/orthoseg/segmodels_keras/blob/main/images/fpn.png

**Backbones**

.. table:: 

    =============  ===== 
    Type           Names
    =============  =====
    VGG            ``'vgg16' 'vgg19'``
    ResNet         ``'resnet50' 'resnet101' 'resnet152'``
    ResNetV2       ``'resnet50v2' 'resnet101v2' 'resnet152v2'``
    DenseNet       ``'densenet121' 'densenet169' 'densenet201'`` 
    Inception      ``'inceptionv3' 'inceptionresnetv2'``
    MobileNet      ``'mobilenet' 'mobilenetv2'``
    EfficientNet   ``'efficientnetb0' 'efficientnetb1' 'efficientnetb2' 'efficientnetb3' 'efficientnetb4' 'efficientnetb5' 'efficientnetb6' 'efficientnetb7'``
    EfficientNetV2 ``'efficientnetv2m'``    
    =============  =====

.. epigraph::
    All backbones have weights trained on 2012 ILSVRC ImageNet dataset (``encoder_weights='imagenet'``). 


Installation
~~~~~~~~~~~~

**Requirements**

1) python 3
2) keras >= 2.10.0 or tensorflow >= 2.10

**PyPI stable package**

.. code:: bash

    $ pip install -U segmodels-keras

**PyPI latest package**

.. code:: bash

    $ pip install -U --pre segmodels-keras

**Source latest version**

.. code:: bash

    $ pip install git+https://github.com/orthoseg/segmodels_keras
    
Documentation
~~~~~~~~~~~~~
Latest **documentation** is avaliable on `Read the
Docs <https://segmodels_keras.readthedocs.io/en/latest/>`__

Change Log
~~~~~~~~~~
To see important changes between versions look at CHANGELOG.md_

Citing
~~~~~~~~

.. code::

    @misc{Yakubovskiy:2019,
      Author = {Pavel Iakubovskii},
      Title = {Segmentation Models},
      Year = {2019},
      Publisher = {GitHub},
      Journal = {GitHub repository},
      Howpublished = {\url{https://github.com/qubvel/segmentation_models}}
    } 

License
~~~~~~~
Project is distributed under `MIT Licence`_.

.. _CHANGELOG.md: https://github.com/orthoseg/segmodels_keras/blob/main/CHANGELOG.md
.. _`MIT Licence`: https://github.com/orthoseg/segmodels_keras/blob/main/LICENSE
