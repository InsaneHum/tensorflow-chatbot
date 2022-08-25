import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
from tensorflow.python.framework import ops
import json
import pickle
import random

# nltk.download('punkt')

stemmer = LancasterStemmer()

with open('intents.json') as f:
    data = json.load(f)

try:
    # load the data for the model
    with open('chatbot_data.pickle', 'rb') as f:  # rb = read bytes
        words, labels, training, output = pickle.load(f)

except:

    words = []
    labels = []
    docs_x = []
    docs_y = []

    # 'stem' the data, i.e. strip data to its root word: etc 'there?' to 'there', 'whats' to 'what':

    for intent in data['intents']:
        for pattern in intent['patterns']:
            wrds = nltk.word_tokenize(pattern)  # tokenize the patterns by extracting all the words into a list
            words.extend(wrds)

            # classify each pattern to correspond with its tag
            docs_x.append(wrds)
            docs_y.append(intent['tag'])

            if intent['tag'] not in labels:
                labels.append(intent['tag'])

    # print(words)
    # print(labels)
    # print(docs_x)
    # print(docs_y)

    # stem and remove duplicate elements
    words = [stemmer.stem(w.lower()) for w in words if w != '?']
    words = sorted(list(set(words)))
    labels = sorted(labels)

    # create "bag of words" list that contains a 1 or 0 depending if the word appears in a sentence
    training = []
    output = []

    # create bag of words for labels
    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w) for w in doc]

        # append 1 if the word exists in the pattern(sentence) and 0 if it does not
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        # set the value of the label to 1 in the bag of words label
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    # convert the 2d lists to numpy arrays
    training = np.array(training)
    output = np.array(output)

    # save the data for the model
    with open('chatbot_data.pickle', 'wb') as f:  # wb = write bytes
        pickle.dump((words, labels, training, output), f)

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
try:
    model.load('chatbot.tflearn')

except:
    print('Failed to load model')
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    # batch size = how many data to train each time
    # epochs = the number of times the model sees the same data
    # show_metric = show formatted output

    # save the model
    model.save('chatbot.tflearn')


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
