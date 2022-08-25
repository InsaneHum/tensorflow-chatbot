import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
from tensorflow.python.framework import ops
import json
import pickle
import random

stemmer = LancasterStemmer()

with open('intents.json') as f:
    data = json.load(f)

# load the data for the model
with open('chatbot_data.pickle', 'rb') as f:  # rb = read bytes
    words, labels, training, output = pickle.load(f)


# create model
ops.reset_default_graph()  # reset tensorflow graph data

# define the input shape that is expected for the model, which is the number of elements in the bag of words
net = tflearn.input_data(shape=[None, len(training[0])])
# add fully connected hidden layer connecting with the input data with 8 nodes to the neural network
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
# define number of nodes for the output layer, with softmax giving a probability of which node is the correct answer
net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
net = tflearn.regression(net)

# train model, DNN = deep neural network
model = tflearn.DNN(net)

# try to load model

model.load('chatbot.tflearn')


# turn user input into bag of words for the model to load
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)


def chat():
    print("Start talking with the bot! (type 'quit' to quit)")
    while True:
        inp = input('You: ')
        if inp.lower() == 'quit':
            break

        # wrap the input int a list as the predict function only works with lists
        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = np.argmax(results)  # argmax returns the index of the label with the highest probability
        tag = labels[results_index]

        # if the model gives an answer with a probability above 0.7, return a response
        if results[results_index] > 0.8:

            for tg in data['intents']:
                if tg['tag'] == tag:
                    responses = tg['responses']

            print(random.choice(responses))

        else:
            print('Sorry, I don\'t understand')


chat()
