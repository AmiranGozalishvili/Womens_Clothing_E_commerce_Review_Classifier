import spacy
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nlp_lem = spacy.load('en_core_web_sm')


def stemming(text):
    """
    Strip suffixes from a word and return the stem.
    It is inefficient to have the algorithm process multiple formats of a word.
    Ex: loved, loving, or loves ===> love
    """
    # creating an object of the PorterStemmer class
    ps = PorterStemmer()

    stemmed_review = [ps.stem(word) for word in text.split(' ') if word not in set(stopwords.words("english"))]
    return ' '.join(stemmed_review)

def lemmentization(text):
    """
    Lemmatization is the grouping together of different forms of the same word.
    it usually refers to doing things properly with the use of a vocabulary and
    morphological analysis of words, normally aiming to remove inflectional endings
    only and to return the base or dictionary form of a word, which is known as the lemma .
    """
    doc = nlp_lem(u"{}".format(text))
    output = " ".join([token.lemma_ for token in doc])
    return output
