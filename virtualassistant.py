import torch

import re
import random

import webbrowser
from datetime import datetime
from deep_translator import GoogleTranslator

from modules import geo
from modules import speech_recognizer as sr

from utils.model import NeuralNet
from utils.nltk_utils import tokenize, bag_of_words

class VirtualAssistant():
    def __init__(self, names: list[str], model_path: str, answers: dict[str, list[str]], owner_pronouns: str = 'he/him', lang='en-US'):
        print('---Setting up ai---')
        self.names = names # user can call bot of multiples ways
        self.lang = lang
        self.owner_pronouns = owner_pronouns
        self.answers = answers

        self.model_data = torch.load(model_path, weights_only=True)
        self.model = NeuralNet(self.model_data["input_size"], self.model_data["hidden_size"], self.model_data["output_size"])
        self.model.load_state_dict(self.model_data["model_state"])
        self.model.eval()

        self.tags = self.model_data["tags"]
        self.all_words = self.model_data["all_words"]

    def wake_up(self, text: str) -> bool:
        return any(re.search(r'\b' + bot_name + r'\b', text) for bot_name in self.names)
    
    def listen(self) -> str:
        audio = sr.listen(1)
        return sr.voice_to_text(audio, self.lang).lower()

    def speech(self, tag: str) -> str:        
        match tag:
            case 'time' | 'day':
                answer = GoogleTranslator(source='auto', target=self.lang.split('-')[0]).translate(datetime.now().strftime(random.choice(self.answers[tag])))
            case 'weather':
                city = geo.get_location()['city']

                if city == 'Unknown location':
                    answer = GoogleTranslator(source='auto', target=self.lang.split('-')[0]).translate('Sorry but i couldn\'t find your location.')
                else:
                    answer = GoogleTranslator(source='auto', target=self.lang.split('-')[0]).translate(geo.get_weather(city))
            case 'location':
                location = geo.get_location()

                if location["city"] == 'Unknown location':
                    answer = GoogleTranslator(source='auto', target=self.lang.split('-')[0]).translate('Sorry but i couldn\'t find your location.')
                else:
                    answer = GoogleTranslator(source='auto', target=self.lang.split('-')[0]).translate(f'You\'re currently in {location["country"]} on the city of {location["city"]} in the {location["region"]} region.')
            case 'browser':
                answer = self.process_answer(self.answers[tag])
                webbrowser.open('http://google.co.kr', new=2)
            case _:
                answer = self.process_answer(self.answers[tag])
        
        print('Dexter:', answer)
        return answer
    
    def process_answer(self, answers: list[str]):
        answer = random.choice(answers)

        if self.owner_pronouns == 'she/her':
            answer = re.sub(r"\bsir\b", "mrs", answer)
    
        return GoogleTranslator(source='auto', target=self.lang.split('-')[0]).translate(answer)

    def predict(self, text) -> tuple[str, int]:
        tokenized_text = tokenize(GoogleTranslator(source='auto', target='en').translate(text))
        input_tensor = bag_of_words(tokenized_text, self.all_words)
        input_tensor = input_tensor.reshape(1, input_tensor.shape[0])
        input_tensor = torch.from_numpy(input_tensor)

        output = self.model(input_tensor)
        _, predict = torch.max(output, dim=1)
        tag = self.tags[predict.item()]

        probabilities = torch.softmax(output, dim=1)
        probability = probabilities[0][predict.item()]
        return tag, probability