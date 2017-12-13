import re
import nltk
import pprint
import operator
import random
import time
import os
import numpy

import bing_api


def markov_chain(raw_text="", sanitize=True, order=1):
    # Tokenize the text into sentences.
    sentences = nltk.sent_tokenize(raw_text)

    # Tokenize each sentence to words. Each item in 'words' is a list with
    # tokenized words from that list.
    tokenized_sentences = []
    for s in sentences:
        w = nltk.word_tokenize(s)
        tokenized_sentences.append(w)

    is_word = re.compile('\w')
    sanitized_sentences = []
    for sent in tokenized_sentences:
        sanitized = [token for token in sent if is_word.search(token)] + ['.']
        sanitized_sentences.append(sanitized)

    if(sanitize == True) :
        sanitized_sentences = my_sanitize_function(['http'], sanitized_sentences);


    #EXO 1 - week 2 
    # add order 
    transitions = {}
    for i, sentence in enumerate(sanitized_sentences):
        for i in range(len(sentence)-order):
            pred = (''.join(x+' ' for x in sentence[(i):(i+order)]))[:-1]
            succ = sentence[i+order];
            if pred not in transitions:
                transitions[pred] = {}

            if succ not in transitions[pred]:
                transitions[pred][succ] = 1.0
            else:
                transitions[pred][succ] += 1.0


    # Compute total number of successors for each state
    totals = {}
    for pred, succ_counts in transitions.items():
        totals[pred] = sum(succ_counts.values())

    # Compute the probability for each successor given the predecessor.
    probs = {}
    for pred, succ_counts in transitions.items():
        probs[pred] = {}
        for succ, count in succ_counts.items():
            probs[pred][succ] = count / totals[pred]

    return probs


# Exercise 4 - week 1
# the function sanitize tokens to allow english words.
# e.g : words like "1c" are refused
def my_sanitize_function(token_list=[], tokenized_sentences=[]):

    is_word = re.compile('^([A-Z]|[a-z])[a-z]*') 
    sanitized_sentences = []
    for sentence in tokenized_sentences:
        sanitized = [token for token in sentence if ((is_word.search(token)) and (token not in token_list))] + ['.']
        sanitized_sentences.append(sanitized)
    #pprint.pprint(sanitized_sentences)
    return sanitized_sentences


# Exercise 6 - week 1
# Generate a text based on state_transition_probabilities. 
def generate(state_transition_probabilities, length=10, start=None):

    # Transform the data to a cumulative distribution function (cdf)
    cdfs = {}
    for pred, succ_probs in state_transition_probabilities.items():
        items = succ_probs.items()
        # Sort the list by the second index in each item and reverse it from
        # highest to lowest.
        sorted_items = sorted(items, key=operator.itemgetter(1), reverse=True)
        cdf = []
        cumulative_sum = 0.0
        for c, prob in sorted_items:
            cumulative_sum += prob
            cdf.append([c, cumulative_sum])
        cdf[-1][1] = 1.0 # We fix the last because of the possible rounding errors.
        cdfs[pred] = cdf
        #print(pred, cdf)

    # Select the start
    if start == None:
        start = random.choice(list(state_transition_probabilities.keys()))
    elif start not in state_transition_probabilities:
        raise ValueError('the \'start\' is not the dictionary')
    
    # init markov_chain 
    markov_chain = []
    markov_chain.append(start)
    #pprint.pprint(cdfs[markov_chain[-1]])

    while len(markov_chain) < length:
        pred = markov_chain[-1] # Last element of the list
        rnd = random.random() # Random number from 0 to 1
        # if the last element has no succ, we stop.
        if(pred not in cdfs) :
            return ''.join([word+' ' for word in markov_chain])
        
        cdf = cdfs[pred]
        cp = cdf[0][1]
        i = 0
        while rnd > cp:
            i += 1
            cp = cdf[i][1]
        succ = cdf[i][0]
        markov_chain.append(succ)

    # finally we return the raw text generated
    return ''.join([word+' ' for word in markov_chain])


def generate2(probs1, probs2, length=10, start=None):
    
    # Transform the data to a cumulative distribution function (cdf)
    cdfs1 = {}
    for pred, succ_probs in probs1.items():
        items = succ_probs.items()
        # Sort the list by the second index in each item and reverse it from
        # highest to lowest.
        sorted_items = sorted(items, key=operator.itemgetter(1), reverse=True)
        cdf = []
        cumulative_sum = 0.0
        for c, prob in sorted_items:
            cumulative_sum += prob
            cdf.append([c, cumulative_sum])
        cdf[-1][1] = 1.0 # We fix the last because of the possible rounding errors.
        cdfs1[pred] = cdf

    cdfs2 = {}
    for pred, succ_probs in probs2.items():
        items = succ_probs.items()
        # Sort the list by the second index in each item and reverse it from
        # highest to lowest.
        sorted_items = sorted(items, key=operator.itemgetter(1), reverse=True)
        cdf = []
        cumulative_sum = 0.0
        for c, prob in sorted_items:
            cumulative_sum += prob
            cdf.append([c, cumulative_sum])
        cdf[-1][1] = 1.0 # We fix the last because of the possible rounding errors.
        cdfs2[pred] = cdf

    result = ""
    while has_noun(result) == False:
        # Select the start
        if(start == None):
            start = random.choice(list(probs2.keys()))
        elif(start not in probs2):
            raise ValueError('the \'start\' is not the dictionary')

        # init markov_chain 
        markov_chain = []

        for x in nltk.word_tokenize(start):
            markov_chain.append(x)

        while len(markov_chain) < length:
            pred = (''.join(x+' ' for x in markov_chain[-2:]))[:-1]
            rnd = random.random() # Random number from 0 to 1
            # if the last element has no succ, we stop.
            if pred not in cdfs2:
                last_word = nltk.word_tokenize(pred)[-1]
                if last_word not in cdfs1:
                    break
                else :
                    cdf = cdfs1[pred]
                    cp = cdf[0][1]
                    i = 0
                    while rnd > cp:
                        i += 1
                        cp = cdf[i][1]
                    succ = cdf[i][0]
                    markov_chain.append(succ)
            else :
                cdf = cdfs2[pred]
                cp = cdf[0][1]
                i = 0
                while rnd > cp:
                    i += 1
                    cp = cdf[i][1]
                succ = cdf[i][0]
                markov_chain.append(succ)

        # finally we extract the raw text generated
        result = ''.join([word+' ' for word in markov_chain])
        #Otherwise we loop forever
        start = None
    return result


def has_noun(text):
    """ Check if there is a noun is the text.
    :param text: a string
    :return: True if noun find. False Otherwise
    """
    tokens = nltk.word_tokenize(text)
    tag = nltk.pos_tag(tokens)
    for key, value in tag :
        if value == "NN":
            return True
    return False


def has_nnp(text):
    """ Check if there is a pronoun is the text.
    :param text: a string
    :return: True if pronoun find. False Otherwise
    """
    tokens = nltk.word_tokenize(text)
    tag = nltk.pos_tag(tokens)
    for key, value in tag :
        if value == "NNP":
            return True
    return False


def has_vb(text):
    """ Check if there is a verb is the text.
    :param text: a string
    :return: True if verb find. False Otherwise
    """
    tokens = nltk.word_tokenize(text)
    tag = nltk.pos_tag(tokens)
    for key, value in tag :
        if value == "VB":
            return True
    return False


def has_adjp(text):
    """ Check if there is an abjectif is the text.
    :param text: a string
    :return: True if abjectif find. False Otherwise
    """
    tokens = nltk.word_tokenize(text)
    tag = nltk.pos_tag(tokens)
    for key, value in tag :
        if value == "ADJP":
            return True
    return False


def get_noun_in_sentence(text):
    """ Find the first noun into a text.
    :param text: a string
    :return: the noun if find inside the text. 
    """
    tokens = nltk.word_tokenize(text)
    tag = nltk.pos_tag(tokens)
    for key, value in tag :
        if value == "NN":
            return key
    return ""


def open_file(path):
    """ load a file into python
    :param text: the path of the file
    :return: the text of the file. 
    """
    file_raw = None
    with open(path, 'r', encoding='utf8') as f:
        file_raw = f.read()
        return re.sub(r'\s+', ' ', file_raw)


def get_sentences_from_lyrics(lyrics_file, nb_setences=3):

    lyrics = open_file(lyrics_file)
    probs1 = markov_chain(lyrics, order=1)

    sentences = []
    for i in range(0, nb_setences):
        sentences.append(generate(probs1, length=3))

    return sentences


def find_images(lyrics_file, nb_imgs=3):
    """ Download images from bing engine. Query are based on lyrics.
    :param lyrics_file: path for lyrics file
    :param nb_imgs: number of images you want to download
    :return: A list of paths for downloaded images.
    """
    new_path = './'+lyrics_file.split('.')[0]+'/'
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    lyrics = open_file(lyrics_file)
    probs1 = markov_chain(lyrics, order=1)

    sentences = []
    timeout = 10000

    # generate sentence based on lyrics
    while len(sentences) < nb_imgs and timeout > 0:
        # move to evaluation
        while True:
            sentence = generate(probs1, length=3)
            if has_noun(sentence) or has_nnp(sentence):
                break
            else:
                timeout -= 1

        if sentence not in sentences:
            sentences.append(sentence)
            timeout = 10000

    paths = []

    count = 1
    pprint.pprint(sentences)
    for sentence in sentences:

        bing_api.get_image(sentence, new_path + str(count))
        time.sleep(3)
        paths.append(new_path+str(count))
        count += 1

    return paths


#generateRethoricalFigure(noun, N=0.2)
