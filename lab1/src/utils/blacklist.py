from nltk.stem import PorterStemmer

# A comprehensive blacklist of common spam trigger words
SPAM_BLACKLIST = {
    # High Pressure / Urgency
    "act now", "urgent", "last chance", "limited time", "immediate", "final call",
    "expire", "asap", "instant", "strictly limited", "get it now",

    # Financial & Scams
    "earn extra cash", "make money", "million dollars", "no investment", "pure profit",
    "financial freedom", "no credit check", "hidden fees", "refinance", "cash bonus",
    "bank account", "wire transfer", "offshore", "lowest price", "save big",

    # Too-Good-To-Be-True / Marketing
    "100% free", "guaranteed", "winner", "you have been selected", "congratulations",
    "risk-free", "money back", "no obligation", "free gift", "free consultation",
    "incredible deal", "prize", "jackpot", "unbelievable",

    # Suspicious / Health / Phishing
    "dear friend", "this is not spam", "no catch", "lose weight fast", "diet",
    "viagra", "pharmacy", "online degree", "social security", "password", 
    "billing address", "verify account", "unsolicited"
}



stemmer = PorterStemmer()
def stem_phrase(phrase):
    return " ".join(stemmer.stem(word) for word in phrase.split())

STEMMED_SPAM_BLACKLIST = {
    stem_phrase(phrase) for phrase in SPAM_BLACKLIST
}