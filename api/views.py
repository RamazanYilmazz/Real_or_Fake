from django.shortcuts import render, redirect
from .models import Image
from .forms import ImageForm
import cv2
from django.http import HttpResponse
from django.conf import settings
from keras.models import load_model
import numpy as np
import io
import h5py
import os
import PIL
from PIL import Image


# Create your views here.

def home(request):
    return render(request, "index.html")

def result(request):
    return render(request, "result.html")



def upload_image(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            # Son yüklenen görseli al
            latest_image = Image.objects.latest('id')
    else:
        form = ImageForm()
    return render(request, 'index.html', {'form': form})


"""
def predict_image(request):
    if request.method == 'POST':
        # formdan resim dosyasını al
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            image_file = form.cleaned_data['image']
            # resim işleme işlemleri
            # ...

            # resim dosyasının URL'sini oluştur
            image_url = settings.MEDIA_URL + image_file.name

            # sonuç sayfasına resim URL'sini gönder
            context = {'image_url': image_url}
            return render(request, 'result.html', context=context)
    else:
        form = ImageForm()
    return render(request, 'upload.html', {'form': form})
"""

def predict_image(request):
    print("predict image çalisiyor")
    #resnet50_model = load_model("my_resnet50_model_100_epochs.h5")
    
    #my_resnet50_model_100_epochs = "my_resnet50_model_100_epochs.h5"
    #resnet50_model = h5py.File(my_resnet50_model_100_epochs)
    
    model_path = os.path.join(settings.STATIC_ROOT, 'my_resnet50_model_100_epochs.h5')
    resnet50_model = load_model(model_path)
    print("modeli aldı")
    if request.method == 'POST':
        # formdan resim dosyasını al
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            #latest_image = Image.objects.latest('id')
            image_file = form.cleaned_data['image']
            print("son görseli aldık")
            
            # resim işleme işlemleri
            image1 = Image.open(image_file)
            image2 = image1.resize((128,128))
            p1 = np.array(image2)
            p1 = p1/255
            p1 = np.expand_dims(p1, axis=0)
            p1.shape
            op = np.argmax(resnet50_model.predict(p1),axis=-1)
            
            if op == [0]:
                print("fake face")
                result = 'Fake Face X'
            else:
                print("real face")
                result = 'Real Face ✓'
            
            
            # resim dosyasının URL'sini oluştur
            image_url = settings.MEDIA_URL + image_file.name
            
            # sonuç sayfasına resim URL'sini ve sonucu gönder
            context = {'image_url': image_url, 'result': result}
            return render(request, 'result.html', context=context)
            
    else:
        form = ImageForm()
    return render(request, 'index.html', {'form': form})

