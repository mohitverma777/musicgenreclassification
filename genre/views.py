from django.shortcuts import render

import os
import pickle
import pandas as pd
from .forms import AudioUploadForm
from .utils import getmetadata  # You will create this in utils.py

# Load pre-trained models and scaler
model_path = os.path.join('models','models.p')
models = pickle.load(open(model_path, 'rb'))
scaler = models['norma']
knn = models['knn']
svm_clf = models['svmp']
rf = models['rf']
dt = models['dt']
lgn = models['lgn']


def classify_audio(request):
    if request.method == 'POST':
        form = AudioUploadForm(request.POST, request.FILES)
        if form.is_valid():
            audio_file = request.FILES['file']

            # Process the uploaded audio file using librosa
            metadata = getmetadata(audio_file)

            if metadata:
                # Convert metadata to DataFrame
                metadata_df = pd.DataFrame([list(metadata.values())], columns=metadata.keys())
                scaled_data = scaler.transform(metadata_df)

                # Make predictions
                genre_prediction_rf = lgn[rf.predict(scaled_data)[0]]
                genre_prediction_dt = lgn[dt.predict(scaled_data)[0]]
                genre_prediction_knn = lgn[knn.predict(scaled_data)[0]]
                genre_prediction_svm = lgn[svm_clf.predict(scaled_data)[0]]
                

                knn_genre = genre_prediction_knn
                svm_genre = genre_prediction_svm
                rf_genre = genre_prediction_rf
                dt_genre = genre_prediction_dt

                # Render the results in the template
                return render(request, 'genre/results.html', {
                    'knn_genre': knn_genre,
                    'svm_genre': svm_genre,
                    'rf_genre': rf_genre,
                    'dt_genre': dt_genre,
                    'metadata': metadata
                })
    else:
        form = AudioUploadForm()

    return render(request, 'genre/upload.html', {'form': form})

