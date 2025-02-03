import speech_recognition as sr

recognizer = sr.Recognizer()
recognizer_listening = False

def listen(adjust_ambient_noise_duration: int = 1) -> sr.AudioData:
    global recognizer_listening
    
    with sr.Microphone() as microphone:
        print('Adjusting for ambient noise, please wait...')
        recognizer.adjust_for_ambient_noise(microphone, adjust_ambient_noise_duration)
        print('Say something now...')
        recognizer_listening = True
        audio: sr.AudioData = recognizer.listen(microphone)
        recognizer_listening = False
    
    return audio

def listening() -> bool:
    return recognizer_listening

def voice_to_text(audio: sr.AudioData, lang: str = 'en-US') -> str:
    try:
        return recognizer.recognize_google(audio, language=lang)
    except sr.UnknownValueError:
        print('Software couldn\'t understand you.')
    except sr.RequestError as e:
        print(f'Software could\'t request translation: {e}')
    
    return ''