import random
import json
import pickle
import numpy as np
import tensorflow as tf
from spellchecker import SpellChecker

import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# Load intents from the JSON file
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
model = tf.keras.models.load_model('chatbot_model.h5')

# Create a spell checker instance
spell = SpellChecker()

def cleanup_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]

    # Use spell checker to correct misspelled words
    corrected_words = [spell.correction(word) for word in sentence_words]

    return corrected_words

def bag_of_words(sentence):
    sentence_words = cleanup_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    for intent in intents_json['intents']:
        if intent.get('tags') == tag:
            return intent['response']
    return "I'm sorry, I don't understand your question."

print("Chatbot is working")

while True:
    message = input("You: ")
    if message.lower() in ["quit", "exit"]:
        break
    # Correct misspelled words in the user's query
    corrected_message = ' '.join(cleanup_sentence(message.lower()))
    classified_intents = predict_class(corrected_message)
    response = get_response(classified_intents, intents)
    print("Bot:", response)
