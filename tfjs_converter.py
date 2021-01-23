import tensorflowjs as tfjs
import tensorflow as tf

keras_model = tf.keras.models.load_model('./keras_model/fer2013.h5')

tfjs.converters.save_keras_model(keras_model, './tfjs_files')
