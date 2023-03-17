import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

new_model = tf.keras.models.load_model('./sacuvan_model')
from keras.utils import image_dataset_from_directory



Xval = image_dataset_from_directory("./sliketest",
                                    image_size=(64,64),
                                    )

classes = Xval.class_names

pred=new_model.predict(Xval)
class_indices = {'Bengal': 0, 'Birman': 1, 'Bombay': 2,'Russian Blue': 3,'Tuxedo': 4 }

predicted_class = np.argmax(pred, axis=1)

predicted_class_names = [list(class_indices.keys())[list(class_indices.values()).index(i)] for i in predicted_class]

print(predicted_class_names)


print("Tačnost na test skupu:", new_model.evaluate(Xval)[1] * 100, '%.')

