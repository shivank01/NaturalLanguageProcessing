from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_sent="This is a sample sentence, showing off the stop words filtration."

stopwords=set(stopwords.words('english'))

word_tokens=word_tokenize(example_sent)

filtered_sentence=[w for w in word_tokens if not w in stopwords]

print(word_tokens)
print(filtered_sentence)