import os
import asyncio
import json
import random
import webbrowser

import python_weather
from datetime import datetime

from modules import voice
from modules.speech_recgonizer import capture_voice_input, convert_voice_to_text

import torch
from utils.model import NeuralNet
from utils.nltk_utils import bag_of_words, tokenize

from deep_translator import GoogleTranslator


INPUT_LANG = 'en-US'
OUTPUT_LANG = 'en'
# OUTPUT_PRONOUNS = 'he/him'
MODEL_FILE = 'model.pth'
BOT_LANG = 'en'
BOT_NAMES = [ 'dex', 'dexter', 'daddy' ]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('assets/intents.json') as file:
    intents = json.load(file)

data = torch.load(MODEL_FILE, weights_only=True)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

async def get_weather(city: str) -> str:
    async with python_weather.Client() as client:
        weather = await client.get(city)
        return (f'Today the weather is {weather.description}, '
                f'temperatures in {weather.temperature}Cº, '
                f'the wind speed is {weather.wind_speed}km/h '
                f'and the air humidity is in {weather.humidity}%.')

async def main() -> None:
    voice.set_voice(0)
    voice.set_voice_rate(160)
    voice.set_voice_volume(1.0)
    
    while True:
        voice_audio = capture_voice_input()
        text_format = convert_voice_to_text(voice_audio, INPUT_LANG).lower()
        # try:
        #     text_format = input('You: ').lower()
        # except (EOFError, KeyboardInterrupt):
        #     break

        if not any((name in text_format) for name in BOT_NAMES):
            continue

        translated_text = GoogleTranslator(source='auto', target=BOT_LANG).translate(text_format)
        # print('You:', translated_text)

        tokenized_text = tokenize(translated_text)
        input_tensor = bag_of_words(tokenized_text, all_words)
        input_tensor = input_tensor.reshape(1, input_tensor.shape[0])
        input_tensor = torch.from_numpy(input_tensor)

        output = model(input_tensor)
        _, predict = torch.max(output, dim=1)
        tag = tags[predict.item()]
        print(f'Predict: {tag}')

        probabilities = torch.softmax(output, dim=1)
        probabilit = probabilities[0][predict.item()]
        print(f'Confidence thereshold: {(probabilit*100):.2f}%')

        if probabilit.item() < 0.75:
            answer = random.choice(intents["replies"]["fallback"])
            answer = GoogleTranslator(source='auto', target=OUTPUT_LANG).translate(answer)
            print('Dexter:', answer)
            voice.speak(answer)
            continue

        match tag:
            case 'exit':
                answer = random.choice(intents["replies"][tag])
                answer = GoogleTranslator(source='auto', target=OUTPUT_LANG).translate(answer)
                print('Dexter:', answer)
                voice.speak(answer)
                break
            case 'time' | 'day':
                answer = datetime.now().strftime(random.choice(intents["replies"][tag]))
            case 'weather':
                answer = await get_weather('Lisbon')
            case 'browser':
                answer = random.choice(intents["replies"][tag])
                webbrowser.open('http://google.co.kr', new=2)
            case _:
                answer = random.choice(intents["replies"][tag])
        
        answer = GoogleTranslator(source='auto', target=OUTPUT_LANG).translate(answer)
        print('Dexter:', answer)
        voice.speak(answer)

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())