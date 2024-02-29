### Data Augmentation ###
import nltk
from nltk.corpus import wordnet
import random
import pandas as pd
import numpy as np


# Function to get synonyms of a word
def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return synonyms

# Function to replace words with synonyms in a sentence
def replace_with_synonyms(sentence):
    tokens = sentence.split()
    augmented_tokens = []
    for token in tokens:
        synonyms = get_synonyms(token)
        if synonyms:
            synonym = random.choice(synonyms)
            augmented_tokens.append(synonym)
        else:
            augmented_tokens.append(token)
    return ' '.join(augmented_tokens)

# Augment the training data by replacing words with synonyms

def synonim_aug(data,class_='Science',multiples = 4):
    # print(data)
    itx_science = data[data.label == class_]
    X_train_augmented = []
    for i in range(multiples):
        X_train_augmented.append([replace_with_synonyms(sentence) for sentence in data.loc[itx_science.index,'name']])
    
    augmented_science = pd.DataFrame(np.concatenate(X_train_augmented))
    augmented_science.columns = ['name']
    return augmented_science