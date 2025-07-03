from django import forms

class AudioUploadForm(forms.Form):
    file = forms.FileField(label="Upload an audio file")