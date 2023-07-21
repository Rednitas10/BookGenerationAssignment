import random
import string
from collections import defaultdict

# Define a function to return a new defaultdict
def default_factory():
    return defaultdict(int)

class MarkovModel:
    def __init__(self):
        """Initializes the Markov Model with an empty default dictionary (Hashtable under the hood). 
        The structure is a dictionary where each key is a tuple of two words, 
        and the value is another dictionary mapping a following word to its count."""

        # Use the function defined above as the default factory
        self.model = defaultdict(default_factory)

    def preprocess_text(self, text):
        """Takes a string of text as input and prepares it for the model by converting it to lowercase,
        removing punctuation, and splitting it into a list of words."""

        text = text.lower()

        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)

        words = text.split()
        return words

    def train(self, text):
        """Takes a string of text as input, preprocesses it, and then updates the model
        based on the word trigrams in the text."""

        words = self.preprocess_text(text)

        for i in range(len(words) - 2):
            w1, w2, w3 = words[i], words[i + 1], words[i + 2]
            self.model[(w1, w2)][w3] += 1

    def generate_next_word(self, w1, w2):
        """Generates a next word given two previous words w1 and w2. The next word is chosen
        randomly from the possible next words in the model, with each word being weighted by
        its count."""

        possible_words = list(self.model[(w1, w2)].keys())

        if not possible_words:
            return None

        word_counts = list(self.model[(w1, w2)].values())
        total_count = sum(word_counts)

        probabilities = [count / total_count for count in word_counts]
        next_word = random.choices(possible_words, probabilities)[0]

        return next_word

    def generate_text(self, num_words):
        #Generates a string of text num_words long based on the model.

        w1, w2 = random.choice(list(self.model.keys()))
        text = [w1, w2]

        for _ in range(num_words):
            next_word = self.generate_next_word(w1, w2)

            if next_word is None:
                break

            text.append(next_word)
            w1, w2 = w2, next_word

        return ' '.join(text)
