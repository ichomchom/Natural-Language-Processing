from spacy.lang.en import English
from collections import Counter

nlp = English()
counts = Counter()
tokenizer = nlp.tokenizer
with open("nyt.txt", "r") as f:
    for line in f:
        tokens = tokenizer(line)
        counts.update([token.text for token in tokens])
print(counts.most_common(10))