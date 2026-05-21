from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np


model = load_model('best_model.keras')


img = image.load_img('znak1.png', target_size=(48,48))
img_array = image.img_to_array(img)


img_array = np.expand_dims(img_array, axis=0)
im_array = img_array / 255.0

prediction = model.predict(img_array)


predicted_class = np.argmax(prediction)

print('Predvidena klasa:', predicted_class)



