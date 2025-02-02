import pygame, pyttsx3
import json
import numpy as np
from random import randint
from threading import Thread
from virtualassistant import VirtualAssistant

ASSISTENT_NAMES = {'d', 'dex', 'dexter', 'daddy'}
MODEL_FILE = 'model.pth'
CONFIDENCE_THERESHOLD = 0.75
running = True

class TTSThread(Thread):
    def __init__(self):
        super().__init__()
        self.tts_engine = pyttsx3.init()
        self.voices = self.tts_engine.getProperty('voices')
        self.set_voice(1)
        self.tts_engine.runAndWait()

        with open('assets/intents.json') as file:
            intents = json.load(file)
    
        self.ai = VirtualAssistant(ASSISTENT_NAMES, MODEL_FILE, intents["replies"], lang='pt-BR')

        self.daemon = True
        self.start()

    def run(self):
        global running

        self.tts_engine.startLoop(False)
        while running:
            if not self.is_speaking():
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
            
            while not running: # finish speech before exit
                self.tts_engine.iterate()
            
            self.tts_engine.iterate()

        self.tts_engine.endLoop()

    def set_voice(self, voice_id: int) -> None:
        if not (0 <= voice_id < len(self.voices)):
            raise IndexError(f'Voice {voice_id} does not exist')
        
        self.tts_engine.setProperty('voice', self.voices[voice_id].id)
        self.tts_engine.runAndWait() # process events event if the engine looping is not running

    def set_voice_volume(self, volume: float) -> None:
        self.tts_engine.setProperty('volume', volume)
        self.tts_engine.runAndWait()
    
    def set_voice_rate(self, rate: int) -> None:
        self.tts_engine.setProperty('rate', rate)
        self.tts_engine.runAndWait()
    
    def speak(self, text: str) -> None:
        self.tts_engine.say(text)
    
    def is_speaking(self) -> bool:
        return self.tts_engine.isBusy()

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
    window_width, window_height = 860, 640
    window = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption('Dexter - your own virtual assistent')
    window_icon = pygame.image.load('assets/icon.jpg')
    pygame.display.set_icon(window_icon)
    while running or tts_thread.is_speaking():
        for event in pygame.event.get():
            match event.type:
                case pygame.QUIT:
                    running = False
                    tts_thread.speak(None)
        
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