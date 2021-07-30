'''
by Geraldine Bengsch 2021
'''
import os
import glob
import nltk
from nltk.tokenize import WhitespaceTokenizer 
from nltk.util import ngrams
from nltk.collocations import BigramCollocationFinder
import nltk.sentiment
import string
from nltk.corpus import PlaintextCorpusReader
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

nltk.download('punkt')
nltk.download("stopwords", quiet=True)



file_location = os.path.join('transcripts', '*.txt')
print("Files are located at: " , file_location)

#retrieves filenames matching a specified pattern
filenames = glob.glob(file_location)
print("Transcripts in the folder: ", filenames)


for file in filenames:
    with open(file,'r') as txtfile:
        transcript = txtfile.read()
        print("Filename: ", txtfile.name, "\n Basic transcript: ", transcript)
        
        #convert text to lower case
        txt_raw = transcript.lower()
        #remove duplicate whitespaces in the string:
        #split() returns list of all words separated by whitespace; join() takes all items and combines them into a string again using a separator
        txt_raw=" ".join(txt_raw.split())
        print("Filename: ", txtfile.name,txt_raw)

        #NLTK retains whitespace as a token by default; to remove spaces from the corpus, use WhitespaceTokenizer(): 
        # create reference variable 
        whitespace_tk = WhitespaceTokenizer()
        # use tokenize method
        tokens = whitespace_tk.tokenize(txt_raw)
        print("\n Filename: ", txtfile.name,"\n Unfiltered text in tokens:" , tokens)

        # create custom stopword list (adapt for personal needs of transcripts)
        new_stopwords = ["f1", "f2", "sei", "tar", "ant", "m1", "*sei:", "*f1:", "*f2:", "*tar:", "*m1:", "*ant:", "beginn", "end", "female", "male", "media", "utf8"]
        
        
        stop_words = nltk.corpus.stopwords.words("english")
        # add custom list to stopword list
        stopwords_all = stop_words + new_stopwords

        # remove stop words from transcript
        tokens_filtered = [word for word in tokens if word not in stopwords_all]
        print("\n Filename: ", txtfile.name,"\n Filtered text in tokens without stop words: " , tokens_filtered)

        # remove punctuation from transcript
        no_punctuation_tokens = ["".join( j for j in i if j not in string.punctuation) for i in  tokens_filtered]
        print("\n Filename: ", txtfile.name,"\n Tokens without punctuation: " , no_punctuation_tokens)

        # Generate word frequencies in the text
        dist = nltk.FreqDist(no_punctuation_tokens)
        print("\n Filename: ", txtfile.name,"\n Word frequencies: "  , dist)

        # plot the frequency distribution of 30 most frequent words
        dist.plot(30,cumulative=False, title=txtfile.name)
        

        # create a word cloud
        wc = WordCloud(max_words =25).generate(txt_raw)
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.show()


        # N-gram analysis
        bigrams = list(ngrams(no_punctuation_tokens, 2))
        print("\n Filename: ", txtfile.name,"\n List of bigrams, unordered: " , bigrams)

        # n-grams: show the times the bigrams are present in descending order
        finder = BigramCollocationFinder.from_words(no_punctuation_tokens, window_size=2)
        ngram = list(finder.ngram_fd.items())
        ngram.sort(key=lambda item: item[-1], reverse=True)
        print("\n Filename: ", txtfile.name,"\n Ngram sorted in descending order: " , ngram)


        # sentiment analysis
        scores_list=[]
        # helper class
        analyzer = nltk.sentiment.SentimentIntensityAnalyzer()
        scores=analyzer.polarity_scores(transcript)
        scores_list.append(scores)
        print("\n Filename: ", txtfile.name,"\n Sentiment scores: " , scores_list)




################################################################
'''
Begin of alternative structuring of the data

Instead of separating the corpus into words within the individual transcripts, this approach generates a corpus of all the transcripts. It then returns a list of all the words that occur in the corpus, the sentences and the parapgraphs
(The way the transcripts I use are structured means that this approach is not as useful as it could be with better pre-porcessed data )
'''


# create corpus of all files
corpus_root = './transcripts'
corpus = PlaintextCorpusReader(corpus_root, '.*txt')
### Print all File IDs in corpus based on text file names ###
text_list=corpus.fileids()
print(f'Corpus created from: {text_list}')

word = corpus.words()
print("\n Words in the corpus: ", word)

# extract sentences from corpus
sentences = corpus.sents()
print("Sentences in the corpus: ", sentences)

# extract paragraphs from corpus
paragraphs = corpus.paras()
print("\n Paragraphs in the corpus: ", paragraphs)


# Generate word frequencies in the text
dist_all = nltk.FreqDist(word)
print("\n Word frequencies: "  , dist_all)

# convert to lower case
lower_case = nltk.FreqDist([w.lower() for w in dist_all])

#remove stop words from transcript
stop_words = nltk.corpus.stopwords.words("english")
words_filtered = [word for word in lower_case if word not in stop_words]
print(words_filtered)

## Creating FreqDist for whole Bag of words, keeping the 20 most common tokens
common = nltk.FreqDist(dist_all).most_common(20)

# tabular representation of 10 most common tokens_filtered
tabular = dist_all.tabulate(10)


## Conversion to Pandas series via Python Dictionary for easier plotting
## Conversion to Pandas series via Python Dictionary for easier plotting
common = pd.Series(dict(common))

## Setting figure, ax into variables
fig, ax = plt.subplots(figsize=(10,10))

## Seaborn plotting using Pandas attributes + xtick rotation for ease of viewing
all_plot = sns.barplot(x=common.index, y=common.values, ax=ax)
plt.xticks(rotation=30);

# not quite sure what happened to the count (compare tabulate and most common)
plt.show()

# generate a separate plot for filtered words
# create freq list and plot for filtered words_filtered## 
# Creating FreqDist for whole Bag of words, keeping the 20 most common tokens
common_filtered = nltk.FreqDist(words_filtered).most_common(20)


## Conversion to Pandas series via Python Dictionary for easier plotting
## Conversion to Pandas series via Python Dictionary for easier plotting
common_filtered = pd.Series(dict(common_filtered))

## Setting figure, ax into variables
fig, ax = plt.subplots(figsize=(10,10))

## Seaborn plotting using Pandas attributes + xtick rotation for ease of viewing
plot_filtered = sns.barplot(x=common_filtered.index, y=common_filtered.values, ax=ax)
plt.xticks(rotation=30);

# not quite sure what happened to the count (compare tabulate and most common)
plt.show()
