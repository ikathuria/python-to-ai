"""
Preprocess Text
"""

import string
import inflect  # convert digits into words

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def text_lowercase(text):
    """Make the test lowercase.

    Args:
        text (str) : input string.

    Returns:
        lowercased input string.
    """
    return text.lower()


def convert_number(text):
    """Convert digits to words.

    Args:
        text (str) : input string.

    Returns:
        converted input string.
    """
    p = inflect.engine()
    temp_str = text.split()

    new_string = []

    for word in temp_str:
        if word.isdigit():
            temp = p.number_to_words(word)
            new_string.append(temp)

        else:
            new_string.append(word)

    temp_str = " ".join(new_string)
    return temp_str


def remove_punctuation(text):
    """Remove all punctuation.

    Args:
        text (str) : input string.

    Returns:
        lowercased input string.
    """
    text = text.replace("_", " ")
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)


def remove_whitespace(text):
    """Remove all extra whitespace.

    Args:
        text (str) : input string.

    Returns:
        input string without extra whitespace.
    """
    return " ".join(text.split())


def remove_stopwords(text):
    """Remove all stopwords.

    Args:
        text (str) : input string.

    Returns:
        input string without stopwords
    """
    stop_words = set(stopwords.words("english"))
    filtered_text = [word for word in text if word not in stop_words]

    return " ".join(filtered_text)


def lemmatize_text(text):
    """Lemmatize all words of text.

    Args:
        text (str) : input string.

    Returns:
        input string with lemmatized words.
    """
    lemmatizer = WordNetLemmatizer()
    word_tokens = word_tokenize(text)

    filtered_text = [word for word in word_tokens]

    lemmas = [lemmatizer.lemmatize(word) for word in filtered_text]

    return " ".join(lemmas)


def preprocess_pipe(text):
    """
    Combining all preprocessing steps.

    1. Converting to lowercase
    2. Converting digits to words
    3. Remove punctuation and whitespace
    4. Removing default stopwords
    5. Lemmatization

    """
    text = text_lowercase(text)
    text = convert_number(text)
    text = remove_punctuation(text)
    text = remove_whitespace(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)

    return text
