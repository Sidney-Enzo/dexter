import pyttsx3

voice_engine = pyttsx3.init()
voices = voice_engine.getProperty('voices')

def set_voice_volume(volume: float) -> None:
    voice_engine.setProperty('volume', volume)

def set_voice_rate(rate: int) -> None:
    voice_engine.setProperty('rate', rate)

def set_voice(id: int) -> None:
    voice_engine.setProperty('voice', voices[id].id)

def speak(text: str) -> None:
    voice_engine.say(text)
    voice_engine.runAndWait()

if __name__ == "__main__":
    for voice in voices:
        print(voice, voice.id)
        voice_engine.setProperty('voice', voice.id)
        speak("Hello world")
        voice_engine.stop()