|Actions Status| |PyPI version|

.. |Actions Status| image:: https://github.com/orthoseg/segmodels_keras/actions/workflows/tests.yml/badge.svg?branch=main
   :target: https://github.com/orthoseg/segmodels_keras/actions/workflows/tests.yml?query=workflow%3ATests

.. |PyPI version| image:: https://img.shields.io/pypi/v/segmodels-keras.svg
   :target: https://pypi.org/project/segmodels-keras


###############
segmodels_keras
###############

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

- High level API (just two lines of code to create model for segmentation)
- **4** models architectures for binary and multi-class image segmentation
  (including legendary **Unet**)
- **20+** available backbones for each architecture
- All backbones have **pre-trained** weights for faster and better convergence
- Helpful segmentation losses (Jaccard, Dice, Focal) and metrics (IoU, F-score)

Documentation
~~~~~~~~~~~~~
The **documentation** of the current stable version is available here:
`Read the docs, stable <https://segmodels_keras.readthedocs.io/en/stable/>`__.

The lastest version of the documentation (= of the main branch) can be found here:
`Read the docs, latest <https://segmodels_keras.readthedocs.io/en/latest/>`__.

Change Log
~~~~~~~~~~
To see important changes between versions look at CHANGELOG.md_

License
~~~~~~~
Project is distributed under `MIT Licence`_.

.. _CHANGELOG.md: https://github.com/orthoseg/segmodels_keras/blob/main/CHANGELOG.md
.. _`MIT Licence`: https://github.com/orthoseg/segmodels_keras/blob/main/LICENSE
