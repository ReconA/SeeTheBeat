import os
import re
import nltk
import pprint
from operator import itemgetter
import operator
import random
from TheRex import TheRex


def downloadAlice():
    # Download Alice's Adventures in Wonderland if it is not yet present
    alice_file = 'alice.txt'
    alice_raw = None

    if not os.path.isfile(alice_file):
        from urllib import request
        url = 'http://www.gutenberg.org/cache/epub/19033/pg19033.txt'
        response = request.urlopen(url)
        alice_raw = response.read().decode('utf8')
        with open(alice_file, 'w', encoding='utf8') as f:
            f.write(alice_raw)
    else:
        with open(alice_file, 'r', encoding='utf8') as f:
            alice_raw = f.read()

    # Remove the start and end bloat from Project Gutenberg (this is not exact, but
    # easy).
    pattern = r'\*\*\* START OF THIS PROJECT GUTENBERG EBOOK .+ \*\*\*'
    end = "End of the Project Gutenberg"
    start_match = re.search(pattern, alice_raw)
    if start_match:
        start_index = start_match.span()[1] + 1
    else:
        start_index = 0
    end_index = alice_raw.rfind(end)
    alice = alice_raw[start_index:end_index]

    # And replace more than one subsequent whitespace chars with one space
    alice = re.sub(r'\s+', ' ', alice)
    return alice


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
        sanitized_sentences = my_sanitize_function(['[Illustration]', 'http'], sanitized_sentences);


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
    if(start == None):
        start = random.choice(list(state_transition_probabilities.keys()))
    elif(start not in state_transition_probabilities):
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
    while(hasNoun(result) == False) : 
        # Select the start
        if(start == None):
            start = random.choice(list(probs2.keys()))
        elif(start not in probs2):
            raise ValueError('the \'start\' is not the dictionary')

        # init markov_chain 
        markov_chain = []

        for x in nltk.word_tokenize(start):
            markov_chain.append(x)
        #pprint.pprint(cdfs1[markov_chain[-1]])

        while len(markov_chain) < length:
            pred = (''.join(x+' ' for x in markov_chain[-2:]))[:-1]
            rnd = random.random() # Random number from 0 to 1
            # if the last element has no succ, we stop.
            if(pred not in cdfs2) :
                last_word = nltk.word_tokenize(pred)[-1]
                if(last_word not in cdfs1):
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

def hasNoun(text):
    tokens = nltk.word_tokenize(text)
    tag = nltk.pos_tag(tokens)
    for key, value in tag :
        if(value == "NN"):
            return True
    return False

def hasNNP(text):
    tokens = nltk.word_tokenize(text)
    tag = nltk.pos_tag(tokens)
    for key, value in tag :
        if(value == "NNP"):
            return True
    return False

def getNounInSentence(text):
    tokens = nltk.word_tokenize(text)
    tag = nltk.pos_tag(tokens)
    for key, value in tag :
        if(value == "NN"):
            return key
    return ""

#Exercise 4-5
# /!\ WARNING : This function can bug if the noun has no category 
#               inside TheRex API.
def generateRethoricalFigure(noun="", N=0.4):
    tr          = TheRex()
    members     = tr.member(noun)
    categories  = members['categories']
    modifiers   = members['modifiers']
    
    #because N is the X% most releveant 
    N = 1-N

    # get the N% of categories
    categories = sorted(categories.items(), key=lambda x:x[1])
    categories = categories[int(len(categories)*N):]
    category = random.choice(categories)[0].split(':')[1]
    print('CATEGORY :',category)

    # get the N% of modifiers
    modifiers = sorted(modifiers.items(), key=lambda x:x[1])
    modifiers = modifiers[int(len(modifiers)*N):]

    modifier = random.choice(modifiers)[0]
    # modifier = 'slippery'
    print('MODIFIER :' ,modifier)
    
    matching_categories = tr.modifier(modifier, noun)['categories']
    pprint.pprint(matching_categories)

    # select category
    print('TEST', [cat.split(':')[0] for cat in matching_categories.keys() if(cat.split(':')[1] == category)])
    modifier_with_cat = [cat.split(':')[0] for cat in matching_categories.keys() if(cat.split(':')[1] == category)]
    
    # If the categrory list has contains the selected categories... 
    if(len(modifier_with_cat)>0):
        modi    = modifier_with_cat[0][0]
        cat     = modifier_with_cat[0][1]
    else:
        modi_cat = random.choice(list(matching_categories.keys())).split(':')
        modi    = modi_cat[0]
        cat     = modi_cat[1]
    print("NEW MODI_CAT", modi, cat)

    # new query for members of this category
    concept = tr.category(modi, cat)['members']
    if(noun not in concept) :
        print("TARGET IS NOT IN NEW CONCEPT")
    else:
        print("TARGET IS IN NEW CONCEPT")

    # sorted concept by score and keep the highest 10% 
    N = 0.9
    concept = sorted(concept.items(), key=lambda x:x[1])
    concept = concept[int(len(concept)*N):]
   
    # generate the rethorical figures
    for c in concept :
        TARGET  = noun
        PROP    = modifier
        SOURCE  = c[0].replace('_', ' ')
        sentence = '{0} is as {1} as {2}'.format(TARGET, PROP, SOURCE)
        print(sentence)


def open_file(path):
    file_raw = None
    with open(path, 'r', encoding='utf8') as f:
        file_raw = f.read()
        return re.sub(r'\s+', ' ', file_raw)


#Exercise 7 - week 1
# TODO : Change to look for the % of matching between the text and the state_transitions
def likelihood(text, state_transitions_probabilities={}):
    new_state = markov_chain(text)
    state_transitions_probabilities.update(new_state)
    return generate(state_transitions_probabilities, length=len(state_transitions_probabilities))


NUMB = open_file('numb.txt')
NOT_AFRAID = open_file('not_afraid.txt')

probs1 = markov_chain(NOT_AFRAID, order=1)
probs2 = markov_chain(NUMB, order=2)


#noun = getNounInSentence(sentence)

sentences = []

while(len(sentences)<15):

    while(True):
        sentence = generate(probs1, length=3)
        if((hasNoun(sentence)) and (hasNNP(sentence))):
            break
    if(sentence not in sentences):
        sentences.append(sentence)

pprint.pprint(sentences)
#noun = 'snow'
#print( 'NOUN :', noun)
#generateRethoricalFigure(noun, N=0.2)














