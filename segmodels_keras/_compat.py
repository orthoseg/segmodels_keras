import keras
from packaging.version import parse as parse_version

KERAS_GTE_3 = parse_version(keras.__version__) >= parse_version("3.0.0")
