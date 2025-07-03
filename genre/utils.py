import librosa
import numpy as np

def getmetadata(filename):
    try:
        y, sr = librosa.load(filename)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rmse = librosa.feature.rms(y=y)
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spec_rolloff = librosa.feature.spectral_rolloff(y=y + 0.01, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)

        mfcc = librosa.feature.mfcc(y=y, sr=sr)

        metadata_dict = {
            'tempo': tempo[0],
            'chroma_stft': np.mean(chroma_stft),
            'rmse': np.mean(rmse),
            'spectral_centroid': np.mean(spec_centroid),
            'spectral_bandwidth': np.mean(spec_bw),
            'rolloff': np.mean(spec_rolloff),
            'zero_crossing_rate': np.mean(zero_crossing_rate)
        }

        for i in range(1, 21):
            metadata_dict.update({'mfcc' + str(i): np.mean(mfcc[i - 1])})

        return metadata_dict

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None
