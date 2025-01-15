import speech_recognition as sr

recognizer = sr.Recognizer()

def capture_voice_input() -> sr.AudioData:
    with sr.Microphone() as microphone:
        print('Adjusting for ambient noise, please wait...')
        recognizer.adjust_for_ambient_noise(microphone, duration=1)
        print('Listening...')
        audio: sr.AudioData = recognizer.listen(microphone)
    
    return audio

def convert_voice_to_text(audio: sr.AudioData, lang: str = 'en-US') -> str:
    try:
        return recognizer.recognize_google(audio, language=lang)
    except sr.UnknownValueError:
        print('Software couldn\'t detect speech.')
    except sr.RequestError as e:
        print(f'Software could\'t request translation: {e}')
    
    return ""