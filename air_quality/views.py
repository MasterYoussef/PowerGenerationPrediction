# airqualityapp/views.py
from django.shortcuts import render

import pandas as pd


import os
from django.conf import settings
import pandas as pd


from django.shortcuts import render

from django.shortcuts import render


'''
premiere partie :
'''

import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")



def prediction_view(request):
    return render(request, 'prediction.html')



# focntion permettant de predire letat de l'air a laide la fonction de calcule d'index aqi 
def predict_air_quality_view(request):
    # Load your models
    import os
    from django.conf import settings
    
    BASE_DIR = settings.BASE_DIR
    MODEL_DIR = os.path.join(BASE_DIR, 'model')
    
    model_paths = {
        'pls_model': os.path.join(MODEL_DIR, 'pls_model.pkl'),
        'dtr_model': os.path.join(MODEL_DIR, 'dtr_model.pkl'),
        'lr_model': os.path.join(MODEL_DIR, 'lr_model.pkl'),
        'knn_model': os.path.join(MODEL_DIR, 'knn_model.pkl'),
        'rfr_model': os.path.join(MODEL_DIR, 'rfr_model.pkl'),
    }
    import joblib

    pls_model = joblib.load(model_paths['pls_model'])
    dtr_model = joblib.load(model_paths['dtr_model'])
    lr_model = joblib.load(model_paths['lr_model'])
    knn_model = joblib.load(model_paths['knn_model'])
    rfr_model = joblib.load(model_paths['rfr_model'])

    if request.method == 'POST':
        # Retrieve form data
        Daily_Yield = float(request.POST.get('Daily_Yield'))
        Total_yield = float(request.POST.get('Total_yield'))
        ambient_temp = float(request.POST.get('ambient_temp'))
        module_temp = float(request.POST.get('module_temp'))
        irradiation = float(request.POST.get('irradiation'))

        # Create a DataFrame with the data
        data = {
            'DAILY_YIELD': [Daily_Yield],
            'TOTAL_YIELD': [Total_yield],
            'AMBIENT_TEMPERATURE': [ambient_temp],
            'MODULE_TEMPERATURE': [module_temp],
            'IRRADIATION': [irradiation]
        }
        df = pd.DataFrame(data)


        # Make predictions
        prediction_results = {
            'prediction_result_pls': pls_model.predict(df)[0],
            'r2_score_pls': '97.7 %',
            'prediction_result_dtr': dtr_model.predict(df)[0],
            'r2_score_dtr': '98.5 %',
            'prediction_result_lr': lr_model.predict(df)[0],
            'r2_score_lr': '98 %',
            'prediction_result_knn': knn_model.predict(df)[0],
            'r2_score_knn': '86.9 %',
            'prediction_result_rfr': rfr_model.predict(df)[0],
            'r2_score_rfr': '99 %'
        }

        # Render the results to the template
        return render(request, 'prediction_interface2.html', prediction_results)

    return render(request, 'prediction_form.html')





# ctte methode permet de lire le fichier teelcharger
def upload_csv(request):
    if request.method == 'POST' and request.FILES.get('csv_file'):
        csv_file = request.FILES['csv_file']

        # chemin complet pour le fichier dans le répertoire 'media/uploads'
        csv_file_path = os.path.join(settings.MEDIA_ROOT, 'uploads', csv_file.name)

        # Enregistrement du  fich  dans le répertoire 'media/uploads'
        with open(csv_file_path, 'wb+') as destination:
            for chunk in csv_file.chunks():
                destination.write(chunk)

        # Chargement  du fich CSV dans un DataFrame pandas
        df = pd.read_csv(csv_file_path)

        # Renvoyer la prévisualisation du DataFrame
        return render(request, 'preview.html', {'table_html': df.to_html()})

    return render(request, 'upload.html')

#pour l'affichage des donnés sous forma d'une table 

def preview_csv(request):
    if request.method == 'POST' and request.FILES['csv_file']:
        # Récupérer le fichier CSV depuis la requête POST
        csv_file = request.FILES['csv_file']

        # Construire le chemin complet pour le fichier
        csv_file_path = os.path.join('media', 'uploads', csv_file.name)
        
        # Enregistrez le fichier dans le répertoire 'media/uploads/'
        with open(csv_file_path, 'wb+') as destination:
            for chunk in csv_file.chunks():
                destination.write(chunk)

        # Chargement du fichier CSV dans un DataFrame pandas
        df = pd.read_csv(csv_file_path)

        # on va convertir le DataFrame en HTML pour l'affichage dans le modèle
        table_html = df.to_html()

        # Envoyer le HTML à la page de prévisualisation
        return render(request, 'preview.html', {'table_html': table_html})

    # Si la méthode n'est pas POST ou si aucun fichier n'est fourni, revenir à la page d'accueil
    return render(request, 'home.html')






def home(request):
    return render(request, 'home.html')
def upload(request):
    return render(request, 'upload.html')
