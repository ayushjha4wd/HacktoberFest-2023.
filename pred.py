from tensorflow.keras.applications import VGG19, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input, decode_predictions
import numpy as np

# Load the pre-trained VGG19 model pre-trained on ImageNet data
model = VGG19(weights='imagenet')

# Load and preprocess an image for prediction
img_path = 'path/to/your/image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Make predictions
predictions = model.predict(x)

# Decode the predictions
print('Predicted:', decode_predictions(predictions, top=3)[0])
