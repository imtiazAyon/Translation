from collections import Counter
from math import exp, log
import random
import re
import tqdm

def read_file(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
    return lines

def count_word_occurrences(text_data):
    """
    Counts the occurrences of each word in the given text data.

    Args:
        text_data (list): A list of sentences.

    Returns:
        dict: A dictionary where the keys are words and the values are the number of 
            occurrences of each word.
    """
    word_counts = {"<START>":0}
    for sentences in text_data:
        sentences = sentences.split('.')
        for sentence in sentences:
            words = re.findall(r'\w+', sentence)
            word_counts["<START>"] += 1

            for word in words:
                if word in word_counts:
                    word_counts[word] += 1
                else:
                    word_counts[word] = 1
        
    return word_counts

def get_ngrams(data, n:int):
    """
    Generate n-grams from the given data.

    Parameters:
    data (list): A list of strings representing the data.
    n (int): The size of the n-gram (e.g., 2 for bigrams, 3 for trigrams).

    Returns:
    Counter: A Counter object containing the n-grams and their frequencies.
    """
    ngrams = []
    for entry in data:
        sentences = entry.split(".")
        for sentence in sentences:
            words = re.findall(r'\w+', sentence)
            if len(words) == 0:
                continue
            words = ['<START>'] + words + ['<END>']
            for i in range(len(words) - n + 1):
                ngram = tuple(words[i:i+n])
                ngrams.append(ngram)
    return Counter(ngrams)

def create_ngram_model(path, n:int):
    """
    Create an n-gram language model based on the given n-grams and (n-1)-grams counts.

    Parameters:
    data (list): A list of strings representing the data.
    n (int): The size of the n-gram (e.g., 2 for bigrams, 3 for trigrams).

    Returns:
    dict: The n-gram language model, where each n-gram is mapped to its probability.
    """
    data = read_file(path)
    model = {}
    ngrams = get_ngrams(data, n)
    n_minus_one_grams = get_ngrams(data, n-1)
    vocabulary = len(set(word for ngram in ngrams for word in ngram))
    for ngram in ngrams:
        prefix = ngram[:-1]
        model[ngram] = (ngrams[ngram] + 1) / (n_minus_one_grams[prefix] + vocabulary)
    model['unknown'] = 1 / vocabulary
    return model

def compute_log_probability(sentence, model):
    """
    Computes the log probability of a sentence based on a given language model.

    Args:
        sentence (str): The input sentence.
        model (dict): The language model, where each bigram is mapped to its probability.

    Returns:
        float: The log probability of the sentence.
    """
    words = re.findall(r'\w+', sentence)
    prob = 0
    prev = "<START>"
    for word in words:
        prob += log(model.get((prev, word), model['unknown']))
        prev = word
    return prob

def get_vocabulary(data):
    """
    Get the vocabulary from a list of data entries.

    Parameters:
    data (list): A list of data entries.

    Returns:
    set: A set containing all unique words in the data entries.
    """
    vocabulary = set()
    for entry in data:
        words = re.findall(r'(?<=\s)\w+(?=\s)|(?<=\s)\d+(?=\s)', entry)
        for word in words:
            vocabulary.add(word)
    return vocabulary

def expectation_maximization(en_data, sv_data, num_iterations):
    """
    Perform the Expectation-Maximization algorithm to estimate translation probabilities.

    Args:
        en_data (list): List of English sentences.
        sv_data (list): List of Swedish sentences.
        num_iterations (int): Number of iterations for the algorithm.

    Returns:
        dict: A dictionary containing the estimated translation probabilities.

    """
    en_words = get_vocabulary(en_data)
    en_words.add("NULL")
    sv_words = get_vocabulary(sv_data)
    translation_probabilities = {}
    for _ in tqdm.tqdm(range(num_iterations)):
        count_en_sv = {}
        count_en = {}
        l = len(en_data)
        for i in range(l):
            en_words = re.findall(r'(?<=\s)\w+(?=\s)|(?<=\s)\d+(?=\s)', en_data[i])
            en_words.append("NULL")
            sv_words = re.findall(r'(?<=\s)\w+(?=\s)|(?<=\s)\d+(?=\s)', sv_data[i])

            for sv_word in sv_words:
                total = 0
                for en_word in en_words:
                    if (sv_word, en_word) not in translation_probabilities:
                        translation_probabilities[(sv_word, en_word)] = random.random()
                    total += translation_probabilities[(sv_word, en_word)]
                for en_word in en_words:
                    alignment_prob = translation_probabilities[(sv_word, en_word)] / total
                    count_en_sv[(sv_word, en_word)] = count_en_sv.get((sv_word, en_word), 0) + alignment_prob
                    count_en[en_word] = count_en.get(en_word, 0) + alignment_prob

        for tuple in translation_probabilities:
            translation_probabilities[tuple] = count_en_sv[tuple] / count_en[tuple[1]]
    return translation_probabilities

def get_likely_swedish(translation_probabilities, en_word, num_results=None):
    """
    Retrieves the most likely Swedish translations for a given English word from a
    dictionary of translation probabilities.
    
    Parameters:
    - translation_probabilities (dict): A dictionary containing translation probabilities
      as key-value pairs, 
    where the keys are tuples of (swedish_word, english_word) and the values are the 
    probabilities.
    - en_word (str): The English word for which the Swedish translations are to be 
    retrieved.
    - num_results (int, optional): The number of results to return. If not specified, 
    all results will be returned.
    
    Returns:
    - list: A list of tuples containing the most likely Swedish translations for the 
    given English word, sorted in descending order of probability. Each tuple contains 
    the Swedish word and its corresponding probability.
    """
    result = {}
    for tuple in translation_probabilities:
        if tuple[1] == en_word:
            result[tuple[0]] = translation_probabilities[tuple]
    result = list(sorted(result.items(), key=lambda x: x[1], reverse=True))
    if num_results is None:
        return list(result)
    return list(result)[:num_results]

def get_likely_english(translation_probabilities, sv_word, num_results=None):
    """
    Retrieves the most likely English translations for a given Swedish word based 
    on translation probabilities.

    Args:
        translation_probabilities (dict): A dictionary containing translation 
        probabilities for word pairs.
        sv_word (str): The Swedish word for which to find likely English translations.
        num_results (int, optional): The maximum number of results to return. If None,
            returns all results. Defaults to None.

    Returns:
        list: A list of tuples containing the likely English translations and their 
            corresponding probabilities, sorted in descending order of probability.
    """
    result = {}
    for tuple in translation_probabilities:
        if tuple[0] == sv_word:
            result[tuple[1]] = translation_probabilities[tuple]
    result = list(sorted(result.items(), key=lambda x: x[1], reverse=True))
    if num_results is None:
        return list(result)
    return list(result)[:num_results]

def random_search_translatation(sentence, translation_probabilities, en_model):
    """
    Translates a sentence from a source language to English using translation 
    probabilities and an English language model.

    Args:
        sentence (str): The sentence to be translated.
        translation_probabilities (dict): A dictionary containing translation 
        probabilities for each word in the source language.
        en_model (object): The English language model used to compute the log 
        probability of a sentence.

    Returns:
        str: The translated sentence in English.

    Raises:
        ValueError: If the words in the sentence are not available in the translation 
        probabilities dataset.
    """

    words = sentence.split(" ")
    en_words_list = [get_likely_english(translation_probabilities, word, num_results=10) for word in words]
    translated = None
    best_score = None
    for _ in range(100):
        chosen_words = []
        word_log_prob = 0
        for en_word in en_words_list:
            if en_word == []:
                return ''
            probabilities =  [item[1] for item in en_word]
            word = random.choices(en_word, weights=probabilities)[0]
            word_log_prob = word_log_prob + word[1]
            if word[1] == "NULL":
                continue
            chosen_words.append(word[0])
        new_sentence = ' '.join(chosen_words)
        score = compute_log_probability(new_sentence, en_model) + word_log_prob
        if best_score is None or score > best_score :
            best_score = score
            translated = new_sentence
        for _ in range(10):
            random.shuffle(chosen_words)
            new_sentence = ' '.join(chosen_words)
            score = compute_log_probability(new_sentence, en_model)
            if score > best_score:
                best_score = score
                translated = new_sentence
    return translated

def get_baseline_model(english_text_path, swedish_text_path):
    english_corpus = read_file(english_text_path)
    swedish_corpus = read_file(swedish_text_path)
    return expectation_maximization(english_corpus, swedish_corpus, 10)

if __name__ == '__main__':
    english_text_path = ''
    swedish_text_path = ''
    swedish = ""
    model = get_baseline_model(english_text_path, swedish_text_path)
    en_model = create_ngram_model(english_text_path, 2)
    translation = random_search_translatation(swedish, model, en_model)
    print(translation)