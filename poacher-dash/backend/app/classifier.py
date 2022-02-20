import sounddevice as sd
from scipy.io.wavfile import write
import wavio as WV

sd.default.device = 0
# Sampling frequency
freq = 44100
  
# Recording duration
duration = 4

def classify_audio():
    retrieved_audio = retrieve_audio()
    return 0

def retrieve_audio():
    recording = sd.rec(int(duration * freq), samplerate=freq, channels=1)
    # Record audio for the given number of seconds
    sd.wait()
    # write("recording0.wav", freq, recording)
    WV.write("recording1.wav", recording, freq, sampwidth=2)
    print(recording)
    return 'recording1.wav'

def main():
    classify_audio()

if __name__ == "__main__":
    main()
