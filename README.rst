segmodels_keras
=========================

This is a fork of the
`segmentation_models <https://github.com/qubvel/segmentation_models>`__ library by
Pavel Iakubovskii, which is not maintained anymore.

This fork is updated to support Keras 3, and also contains some bug fixes, some
improvements and support for some extra backbone models.

It is not meant as a full replacement of the original library, but rather as a
solution for a library I developed and depended on segmentation_models:
`orthoseg <https://github.com/orthoseg/orthoseg>`__ . Hence, full backwards
compatibility,... or support for all features is not guaranteed.


**The main features** of this library are:

-  High level API (just two lines of code to create model for segmentation)
-  **4** models architectures for binary and multi-class image segmentation
   (including legendary **Unet**)
-  **20+** available backbones for each architecture
-  All backbones have **pre-trained** weights for faster and better
   convergence
- Helpful segmentation losses (Jaccard, Dice, Focal) and metrics (IoU, F-score)

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
