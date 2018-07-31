import PyPDF2
from nltk.tokenize import word_tokenize
from itertools import chain
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer

#reading from the pdf and storing all words in pdfwords
pdfFileObj = open('test.pdf', 'rb')
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
pdfwords = list(chain(*(word_tokenize(pdfReader.getPage(i).extractText()) for i in range(0,pdfReader.numPages))))
pdfFileObj.close()

#cleaning the pdfwords from the previous step by removing stop words and keeping only alpha strings
stop_words = set(stopwords.words('english'))
pdfwords = [word for word in pdfwords if word.lower() not in stop_words and word.isalpha()]

#now stemming words
stemmer = SnowballStemmer("english")
stemmed_words = [stemmer.stem(word) for word in pdfwords]
stemmed_words = [(".").join(stemmed_words)]                       #converting into 1 document

#now using count_vectorizer to count the words
vectorizer = CountVectorizer()
vector = vectorizer.fit_transform(stemmed_words)
vocab = vectorizer.get_feature_names()
vector = vector.toarray()
dictionary = dict(zip(vocab, vector[0]))
sorted_by_value = sorted(dictionary.items(), key=lambda kv: kv[1],reverse = True)
f=open("common_words.txt", "w+")
f.write(("\n").join(map(str,sorted_by_value)))
f.close()