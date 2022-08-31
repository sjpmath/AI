from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

model = load_model('keras_model.h5')
data = np.ndarray(shape=(1,224,224,3), dtype=np.float32)
image = Image.open('turtle.png')
size = (224,224)
image = ImageOps.fit(image,size,Image.ANTIALIAS)
image_array = np.asarray(image)

if image_array.shape[2]>3:
    image_array = image_array[:,:,:3] #make into rgb

normalized_image_array = (image_array.astype(np.float32)/127.0)-1
data[0] = normalized_image_array #pixel data

prediction = model.predict(data) #predict using data
print(prediction)

index2category = {
    0: 'mario',
    1: 'luigi',
    2: 'koopa troopa'
}
for i, prob in enumerate(prediction[0].tolist()):
    print(index2category[i], '{:.4f}'.format(prob))
