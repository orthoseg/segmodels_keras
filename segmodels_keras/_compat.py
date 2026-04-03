import keras
from packaging.version import parse as parse_version

try:
    keras_version = parse_version(keras.__version__)
except Exception:
    import tf.keras as tf_keras

    keras_version = parse_version(tf_keras.__version__)

KERAS_GTE_3 = keras_version >= parse_version("3.0.0")
