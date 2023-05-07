from django.db import models
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from tensorflow import keras
from keras.models import load_model



class Image(models.Model):
    title = models.CharField(max_length=200)
    image = models.ImageField(upload_to='images/')

    class Meta:
        db_table = "api_image"


"""
def predict(request):
    resnet50_model = load_model("my_resnet50_model_100_epochs.h5")
    image = request.FILES.get('image')  # Burada 'image' parametresi, formda yüklenen dosyanın adıdır.
    image2=image.resize((128,128))
    p1 = np.array(image2)
    p1 = p1/255
    p1 = np.expand_dims(p1, axis=0)
    p1.shape
    
    op=np.argmax(resnet50_model.predict(p1),axis=-1)
    print(op)
    if op== [0]:
        print('Fake Face')
    else:
        print("Real Face")
"""        
