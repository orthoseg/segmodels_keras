.. Segmentation Models documentation master file, created by
   sphinx-quickstart on Tue Dec 18 17:37:58 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the segmodels_keras documentation!
===============================================

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


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   user_guide
   api
   support



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
