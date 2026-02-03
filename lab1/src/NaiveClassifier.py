import re

from src.utils.blacklist import STEMMED_SPAM_BLACKLIST

class NaiveSpamClf(object):
    def __init__(self):
        # The actual blacklist of words. Init here for better understanding
        self.blacklist = None

        # The blacklist converted to regex
        self.blacklist_regex = None

    def fit(self, X, y=None):    
        # X, y kept for API compatibility
    
        # Assume SPAM_BLACKLIST is already stemmed
        self.blacklist = list(STEMMED_SPAM_BLACKLIST)

        # Compile regex ONCE
        self.blacklist_regex = re.compile(
            r'\b(?:' + '|'.join(map(re.escape, self.blacklist)) + r')\b',
            flags=re.IGNORECASE
        )


    def predict(self, X):
        # FAST vectorized prediction

        texts = (
            X['subject'].fillna('').astype(str)
            + " "
            + X['message'].fillna('').astype(str)
        )

        # Vectorized regex match
        is_spam = texts.str.contains(self.blacklist_regex, regex=True)

        return is_spam.astype(int).to_numpy()
