from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

corpus = [
		"Preach lol! :) RT @mention: #alliwantis this type of weather all the time.. I live for beautiful days like this! #minneapolis",
		"@mention good morning sunshine","rhode island",
		"RT @mention: I absolutely love thunderstorms!",
		"@mention right this weather is something else",
		"TOP CHOICE --&gt; {link} - Today is awesome!!! Free comic books, lunch with my mama, sunshine & DJ'n ... (via @mention)",
		"CCAk Trail Update: Archangel Road, Mat-Su - 8:00 PM, Thu May 05, 2011: Snow column beginning to break up especia...  {link}"
]

counts = [[3, 0, 1],
          [2, 0, 0],
          [3, 0, 0],
          [4, 0, 0],
          [3, 2, 0],
          [3, 0, 2]
]

# count vectorizer
vectorizer = CountVectorizer(min_df=1)
X = vectorizer.fit_transform(corpus)
X.toarray()

# tfidf transformer
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(counts)
tfidf.toarray()

# combination : tfidf vectorizer
vectorizer = TfidfVectorizer(min_df=1)
X = vectorizer.fit_transform(corpus)
X.toarray()

# hasher : save time and space
hv = HashingVectorizer()
hv.transform(corpus)