
from collections import defaultdict

import numpy as np


class MarkovText(object):

    def __init__(self, corpus):
        self.corpus = corpus
        self.term_dict = None

    def get_term_dict(self):
        # Split the corpus into individual words
        tokens = self.corpus.split()
        transitions = defaultdict(list)

        # Store all possible next words, including duplicates
        for current, next_word in zip(tokens, tokens[1:]):
            transitions[current].append(next_word)

        # Include the final word even if nothing follows it
        if tokens:
            transitions.setdefault(tokens[-1], [])

        self.term_dict = dict(transitions)
        return self.term_dict

    def generate(self, seed_term=None, term_count=15):
        # Build the dictionary if it does not exist
        if self.term_dict is None:
            self.get_term_dict()

        # Check whether the seed term exists
        if seed_term is not None and seed_term not in self.term_dict:
            raise ValueError("The seed term is not in the corpus.")

        if term_count < 0:
            raise ValueError("term_count cannot be negative.")

        if term_count == 0:
            return ""

        if not self.term_dict:
            raise ValueError("Cannot generate text from an empty corpus.")

        # Select a random starting word if no seed is provided
        if seed_term is None:
            seed_term = np.random.choice(list(self.term_dict))

        generated = [seed_term]
        current = seed_term

        words_with_followers = [
            word for word, followers in self.term_dict.items()
            if followers
        ]

        # Generate words using the Markov property
        while len(generated) < term_count:
            followers = self.term_dict[current]

            if followers:
                current = np.random.choice(followers)
            elif words_with_followers:
                # Restart when a word has no possible next word
                current = np.random.choice(words_with_followers)
            else:
                current = seed_term

            generated.append(current)

        return " ".join(generated)
