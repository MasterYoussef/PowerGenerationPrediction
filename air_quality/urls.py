# airqualityapp/urls.py
from django.urls import path
from .views import home,upload,upload_csv
from . import views
from .views import predict_air_quality_view,prediction_view

urlpatterns = [
    path('', home, name='home'),
    path('upload/', upload, name='upload'),
    path('upload/', upload_csv, name='upload_csv'),

    path('preview/', views.preview_csv, name='preview_csv'),
     path('predict_air_quality/', predict_air_quality_view, name='predict_air_quality_view'),
      path('prediction/', prediction_view, name='prediction_view'),



]
