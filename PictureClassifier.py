import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import sys
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# copy from trainer program
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/horse-or-human.zip"
data_dir = tf.keras.utils.get_file(origin=dataset_url, extract=True,cache_subdir='horse-or-human')
data_dir = tf.keras.utils.get_file(origin=dataset_url, extract=True)
data_dir = pathlib.Path(data_dir).with_suffix('')
image_count = len(list(data_dir.glob('*/*.png')))
batch_size = 32
img_height = 300
img_width = 300
# end copy section


train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
class_names = train_ds.class_names
print(class_names)
model=keras.models.load_model("HorseHumanPredictor.keras")
# Classify an image that wasn't included in the training or validation
url = "https://www.example.jpg"
path = tf.keras.utils.get_file('Red_sunflower', origin=url)

file=sys.argv[1]
img = tf.keras.utils.load_img(
    file, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
plt.imshow(img,cmap=plt.cm.binary)
plt.show()