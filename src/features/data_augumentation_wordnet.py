from tqdm import tqdm
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet as wn

def tag(sentence):
    words = word_tokenize(sentence)
    words = pos_tag(words)
    return words

def paraphraseable(tag):
    return tag.startswith('NN') or tag == 'VB' or tag.startswith('JJ')

def pos(tag):
    if tag.startswith('NN'):
        return wn.NOUN
    elif tag.startswith('V'):
        return wn.VERB

def synonyms(word, tag):
    lemma_lists = [ss.lemmas() for ss in wn.synsets(word, pos(tag))]
    lemmas = [lemma.name() for lemma in sum(lemma_lists, [])]
    return set(lemmas)

def synonymIfExists(sentence):
    for (word, t) in tag(sentence):
        if paraphraseable(t):
            syns = synonyms(word, t)
            if syns:
                if len(syns) > 1:
                    yield [word, list(syns)]
                    continue
        yield [word, []]

def paraphrase(sentence):
    return [x for x in synonymIfExists(sentence)]

def sentance_from_synonyms(a):
    sentence = []
    for i in range(len(a)):
        if a[i][1] ==  []:
            sentence.append(a[i][0])
        else:
            sentence.append(a[i][1][0])
    return ' '.join(sentence)

def get_aug_data(data):
    train_formal_aug = []
    train_informal_aug = []
    for i in tqdm(data):
        tmp = sentance_from_synonyms(paraphrase(data['Formal text'][i]))
        train_formal_aug.append(tmp)

        tmp_in = sentance_from_synonyms(paraphrase(data['Informal text'][i]))
        train_informal_aug.append(tmp_in)

    return train_formal_aug, train_informal_aug