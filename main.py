import pygame

from elevenlabs import stream, Voice, VoiceSettings
from elevenlabs.client import ElevenLabs

import json
import numpy as np
from random import randint

from threading import Thread

from conf import * # here is where the api key is hidden
import modules.speech_recognizer as sr
from virtualassistant import VirtualAssistant

MODEL = 'eleven_multilingual_v2'

client = ElevenLabs(api_key=API_KEY)
VOICE = Voice(
    voice_id=VOICE_ID,
    settings=VoiceSettings(
        stability=1,
        similarity_boost= 0.75
    )
)

ASSISTENT_NAMES = {'d', 'dex', 'dexter', 'daddy'}
MODEL_FILE = 'model.pth'
CONFIDENCE_THERESHOLD = 0.75

running = True

class TTSThread(Thread):
    def __init__(self):
        super().__init__()
        with open('assets/intents.json') as file:
            intents = json.load(file)
        
        self.speaking = False
        self.ai = VirtualAssistant(ASSISTENT_NAMES, MODEL_FILE, intents["replies"], lang='pt-BR')

        self.daemon = True
        self.start()

    def run(self):
        global running

        while running:
            text = self.ai.listen()
            print('You:', text)
            # text = input('You: ').lower()
            if not (text and self.ai.wake_up(text)):
                continue

            tag, probability = self.ai.predict(text)
            print(
                '---Dex woke up---',
                f'Predict: {tag}',
                f'Confidence thereshold: {(probability*100):.2f}%;',
                sep='\n'
            )

            if probability < CONFIDENCE_THERESHOLD:
                tag = 'fallback'

            self.speak(self.ai.speech(tag))
            if tag == 'exit':
                running = False
    
    def speak(self, text: str) -> None:
        audio_stream = client.generate(
            text=text,
            voice=VOICE
        )

        self.speaking = True # TEMPORARY
        stream(audio_stream)
        self.speaking = False
    
    def is_speaking(self) -> bool:
        return self.speaking

def draw_wave_line(window: pygame.surface, amplitude: int, color: tuple[int], position: tuple[int], width: int) -> None:
    wave_line = [
        (position[0], position[1]),
        (position[0] + width, position[1])
    ]
    if amplitude >= 5:
        wave_line = [(position[0] + x, position[1] + amplitude*np.sin(x*0.02)) for x in range(width)]
    
    pygame.draw.lines(window, color, False, wave_line, 2)

def main() -> None:
    global running
    tts_thread = TTSThread()

    pygame.init()
    window_icon = pygame.image.load('assets/icon.jpg')
    window_width, window_height = 860, 640
    window = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption('Dexter - your own virtual assistent')
    pygame.display.set_icon(window_icon)
    while running or tts_thread.is_speaking():
        for event in pygame.event.get():
            match event.type:
                case pygame.QUIT:
                    running = False
        
        window.fill('black')
        amplitude = 0
        color = (128, 128, 128)
        if tts_thread.is_speaking():
            amplitude = randint(0, 64)
            color = (255, 255, 255)
        
        draw_wave_line(window, amplitude, color, (0, window_height//2), window_width)
        pygame.display.update()
    
    pygame.quit()

if __name__ == '__main__':
    main()