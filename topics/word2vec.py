import logging
import os
import tempfile
from gensim import corpora, models, similarities
from collections import defaultdict
from pprint import pprint
import string
import re
from nltk.stem import WordNetLemmatizer
import random
import matplotlib.pyplot as plt
import operator
from decimal import Decimal
from nltk import pos_tag
#import nltk

#nltk.download('wordnet')

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

TEMP_FOLDER = tempfile.gettempdir()
print('Folder "{}" will be used to save temporary dictionary and corpus.'.format(TEMP_FOLDER))
print('\n\n')

#corpusSourceToListOfDocuments returns all documents in folder path
#as a list of strings.
#path is the path to the folder containing the corpus
def corpusSourceToListOfDocumets(path):
	corpusL = []
	for filename in os.listdir(path):
		asciiErr = False
		utf8Err = False
		defaultErr = False
		content = None
		fullpath = path + '/' + filename
		print('filename is: ' + filename)
		if(filename[0] != '.'):
			#currentFileAscii = open(fullpath,"r",encoding='ascii')
			#currentFileUtf8 = open(fullpath,"r",encoding='utf-8')
			#currentFileDefult = open(fullpath,"r")
			#ascContent =""
			#utfContent =""
			#defaultContent=""
			t = open(fullpath,'r',errors ='ignore')
			r = t.read()
			content = r
			'''
			try:
				ascContent = currentFileAscii.read()
			except:
				asciiErr=True
			try:
				utfContent = currentFileUtf8.read()
			except:
				utf8Err=True

			try:
				defaultContent = currentFileDefult.read()
			except:
				defaultErr = True

			if asciiErr == False:
				content = ascContent
			elif utf8Err == False:
				content = utfContent
			elif defaultErr == False:
				content = defaultContent
			else:
				currentFileAscii.close()
				currentFileUtf8.close()
				currentFileDefult.close()
				print('error reading file: ' + fullpath)
			'''
			if content != None:
				corpusL.append(content)

			#currentFileAscii.close()
			#currentFileUtf8.close()
			#currentFileDefult.close()

			#f = open(fullpath, 'r')
			#fcontents = f.read()
			#f.close()

	#print(corpusL)
	return corpusL


#tokenize(corpusL) tokenizes each docment in corpusL.
#each document is lowercased split on whitespace, stopwords and words
#conataning only numbers and punctuation are removed.
#corpusL is a list of strings where each string is a document
#this function is for generating the for the corpus dictionary
def tokenize(corpusL):
	#stoplist = getStopWordList('./stopwords.txt')
	#stopSet = set(stoplist)


	#wordnet_lemmatizer = WordNetLemmatizer()

	tokenizedCorpus = []

	#numberRemover = re.compile("[\d{}]+$".format(re.escape(string.punctuation)))

	for document in corpusL:
		wordListUntokenized = document.lower().split()
		wordListTokenized = []

		for word in wordListUntokenized:
			w = word.strip(string.punctuation)
			wordListTokenized.append(w)
			#w = wordnet_lemmatizer.lemmatize(w)
			#if(w not in stopSet and w != '' and not numberRemover.match(w)):
				#wordListTokenized.append(w)
		tokenizedCorpus.append(wordListTokenized)


	#print('DEBUGING')
	#pprint(tokenizedCorpus[0:15])

	return tokenizedCorpus


#======================================================================================================================================================================================================================================
##################################################
#                    MAIN                        #
##################################################



#####	Read and Tokenize corpus 	#####

#read source data


#	!!!!!!!!!!!! change to folder with the corpus
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
corpusL = corpusSourceToListOfDocumets('_input')
print('Corpus Read')

#tokenize corpus
tokenizedC = tokenize(corpusL)
print('\nCorpus Tokenized')

print('Begining to train models')

##### Train Word2Vec  #####
#word2vec (https://radimrehurek.com/gensim/models/word2vec.html)
model = models.Word2Vec(
        tokenizedC,
        size=150,
        window=10,
        min_count=2,
        workers=3)
model.train(tokenizedC, total_examples=len(tokenizedC), epochs=10)

#save model
model.save("word2vec.model")


##################################################
#                    END                        #
##################################################
#======================================================================================================================================================================================================================================










'''
class MyCorpus(object):
	@staticmethod
	def createCorpus(docs):
		corp = MyCorpus()
		corp.docL = docs
		return corp

	def __init__(self):
		#length = 0
		docL = []

	def __iter__(self):
		for doc in self.docL:
			yield dictionary.doc2bow(MyCorpus.tokenizeDoc(doc))

	def __len__(self):
		return len(self.docL)

	#returns list of tokens in lowercase split on whitespace with punctuation removed, lemmatized
	@staticmethod
	def tokenizeDoc(doc):
		wordnet_lemmatizer = WordNetLemmatizer()
		docT = []
		for word in doc.lower().split():
			w = word.strip(string.punctuation)
			w = wordnet_lemmatizer.lemmatize(w)
			if(w != ''):
				docT.append(w)

		return docT


	@staticmethod
	def posTokenizeDoc(doc):
		wordnet_lemmatizer = WordNetLemmatizer()
		docT = []
		wordLSplit= doc.lower().split()
		#print("hiiiiiiiiiiii")
		wordListPunctuationRemoved = []
		for word in wordLSplit:
			w = word.strip(string.punctuation)
			if(w != ''):
				wordListPunctuationRemoved.append(w)
		#print('heeeeeeeeeeeeeey')
		wordLTagged = []
		#pprint(wordListPunctuationRemoved)
		wordLTagged = pos_tag(wordListPunctuationRemoved)
		#print('hhhhhhhhhhh')
		for wT in wordLTagged:
			if('NN' in wT[1]):
				docT.append(wordnet_lemmatizer.lemmatize(wT[0], pos='n'))
			elif('VB' in wT[1]):
				docT.append(wordnet_lemmatizer.lemmatize(wT[0], pos='v'))

		return docT


####### END CLASS #######
'''


'''
def posTokenize(corpusL):
	stoplist = getStopWordList('./stopwords.txt')
	stopSet = set(stoplist)

	wordnet_lemmatizer = WordNetLemmatizer()

	tokenizedCorpus = []

	numberRemover = re.compile("[\d{}]+$".format(re.escape(string.punctuation)))

	for document in corpusL:
		wordListUntokenized = document.lower().split()
		wordListTokenized = []


		wordListPunctuationRemoved = []
		for word in wordListUntokenized:
			w = word.strip(string.punctuation)
			#wOld = w
			#w = wordnet_lemmatizer.lemmatize(w)
			#print(wOld + " to " + w)
			if((w not in stopSet) and (w != '') and not numberRemover.match(w)):
				wordListPunctuationRemoved.append(w)

		wordListTagged = pos_tag(wordListPunctuationRemoved)
		for wT in wordListTagged:
			if('NN' in wT[1]):
				wordT = wordnet_lemmatizer.lemmatize(wT[0], pos='n')
			elif('VB' in wT[1]):
				wordT = wordnet_lemmatizer.lemmatize(wT[0], pos='v')
			else:
				wordT = ''

			if(wordT != '' and wordT not in stopSet):
				wordListTokenized.append(wordT)


		tokenizedCorpus.append(wordListTokenized)
		#tokenizedCorpus.append(wordListPunctuationRemoved)


	#print('DEBUGING')
	#pprint(tokenizedCorpus[0:15])

	return tokenizedCorpus

#getStopWordList(pathToSWL) returns a list of stopwords contained in file
#located at pathToSWL.
#getStopWordList expects the stop words in the file to be deliniated by whitespace
def getStopWordList(pathToSWL):
	print('Reading stopword list from: ' + pathToSWL + '\n')

	f = open(pathToSWL, 'r')
	fcontents = f.read()
	swl = []
	swl = fcontents.split()
	f.close()

	return swl


def generateTopicKeywordList(model, numTopics, numKeywords):
	topicList = []
	i = 0
	while(i < numTopics):
		currTopic = model.show_topic(i, topn=numKeywords)
		topicI = []
		for keyW in currTopic:
			topicI.append(keyW[0])

		topicList.append(topicI)
		i += 1
	return topicList

def KeywordListToCSV(keywordList, modelType):

	numTopics = len(keywordList)
	numKeywords = len(keywordList[0])

	csv = open(modelType + '.csv', 'w+')
	print('Writing File: ' + modelType + '.csv')

	i = 0
	while(i < numTopics):
		if(i == (numTopics - 1)):
			csv.write('"' + 'Topic ' + str(i + 1) + '"\n')
			break
		else:
			csv.write('"' + 'Topic ' + str(i + 1) + '",')
		i += 1

	i = 0
	j = 0
	while(j < numKeywords):
		i = 0
		while(i < numTopics):
			if(i == (numTopics - 1)):
				csv.write('"' + keywordList[i][j] + '"\n')
				break
			else:
				csv.write('"' + keywordList[i][j] + '",')
			i += 1
		j += 1


	csv.close()

#code for evaluation LDA model

#randomly selects n documents to remove from the corpus before
#the model is trained, for validation.
#returns list of heled out documents.
#THIS FUNCTION MODIFIES corpusL
def generateHoldOutList(corpusL, n):
	randSet = set()
	corpusLen = len(corpusL)
	i = 0
	while(len(randSet) < n):
		randSet.add(random.randint(0, corpusLen - 1))
		i += 1

	testCorpus = []
	randList = list(randSet)
	randList.sort(reverse=True)
	#build test document set
	i = 0
	while(i < len(randList)):
		testCorpus.append(corpusL[randList[i]])
		del corpusL[randList[i]]
		i += 1

	return testCorpus

def evaluateLDANumTopicsGraph(dictionary, corpus, texts, limit):
	"""
	Function to display num_topics - LDA graph using c_v coherence

	Parameters:
	----------
	dictionary : Gensim dictionary
	corpus : Gensim corpus
	limit : topic limit

	Returns:
	-------
	lm_list : List of LDA topic models
	"""
	c_v = []
	lm_list = []
	for num_topics in range(1, limit + 1):
		print('\n===========================     Training on ' + str(num_topics) + ' topics     ===========================\n')
		lm = models.LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary,  passes=10, update_every=2, iterations=250, alpha='auto', eta='auto')
		lm_list.append(lm)
		cm = models.CoherenceModel(model=lm, texts=texts, dictionary=dictionary, coherence='c_v')
		c_v.append(cm.get_coherence())

	#pprint(c_v)

	# Show graph
	x = range(1, limit + 1)
	plt.plot(x, c_v)
	plt.xlabel("Number of Topics")
	plt.ylabel("Coherence Score")
	plt.legend(("c_v"), loc='best')
	plt.show()

	return (lm_list, c_v)


def evaluateLSINumTopicsGraph(dictionary, corpus, texts, limit):
	"""
	Function to display num_topics - LDA graph using c_v coherence

	Parameters:
	----------
	dictionary : Gensim dictionary
	corpus : Gensim corpus
	limit : topic limit

	Returns:
	-------
	lm_list : List of LDA topic models
	"""
	c_v = []
	lm_list = []
	for num_topics in range(1, limit + 1):
		print('\n===========================     Training on ' + str(num_topics) + ' topics     ===========================\n')
		lm = models.LsiModel(corpus, id2word=dictionary, num_topics=num_topics, power_iters=50, extra_samples=5)
		lm_list.append(lm)
		topics = []
		for topic_id, topic in lm.show_topics(num_topics=num_topics, formatted=False):
			topic = [word for word, _ in topic]
			topics.append(topic)
		cm = models.CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence='c_v')
		c_v.append(cm.get_coherence())

	#pprint(c_v)

	# Show graph
	x = range(1, limit + 1)
	plt.plot(x, c_v)
	plt.xlabel("Number of Topics")
	plt.ylabel("Coherence Score")
	plt.legend(("c_v"), loc='best')
	plt.show()

	return (lm_list, c_v)

#shows table of topNkeywords of best topNtopics for topic model tm, uses coherence to evaluate topic
def showBestTopics(tm, texts, dictionary, windowSize, topNTopics, topNKeywords):
	coherence_values = {}
	for n, topic in tm.show_topics(num_topics=tm.num_topics, formatted=False):
		topic = [word for word, _ in topic]
		cm = models.CoherenceModel(topics=[topic], texts=texts, dictionary=dictionary, window_size=windowSize)
		coherence_values[n] = cm.get_coherence()

	coherence_values = sorted(coherence_values.items(), key=operator.itemgetter(1), reverse=True)


	#bestTopicList is a list of lists of the form [topic number, topic coherence, keyword 1, keyword 2, ..., keyword n])
	bestTopicList = []
	for i in range(0, topNTopics):
		print('topic number: ' + str(coherence_values[i][0]) + ', topic coherence: ' + str(coherence_values[i][1]))

		L = []
		L.append(coherence_values[i][0])
		L.append(round(Decimal(coherence_values[i][1]), 3))

		kwL = tm.show_topic(coherence_values[i][0], topn=topNKeywords)

		keywordOnly = []
		for k in kwL:
			keywordOnly.append(k[0])

		L.extend(keywordOnly)
		bestTopicList.append(L)

	#draw table
	colLabels = ['Topic Number', 'Coherence']
	i = 0
	while(i < (len(bestTopicList[0]) - 2)):
		colLabels.append('Keyword ' + str(i+ 1))
		i += 1

	widths = [0.06, 0.05]
	i = 0
	while(i < (len(bestTopicList[0]) - 2)):
		widths.append(0.08)
		i += 1


	fig, ax = plt.subplots(figsize=(15,8))
	fig.patch.set_visible(False)
	ax.axis('off')
	ax.axis('tight')
	#plt.figure(figsize=(8, 4), dpi=120)
	tbl = ax.table(cellText=bestTopicList, colLabels=colLabels, cellLoc='center', loc='lower left', colWidths=widths)
	tbl.auto_set_font_size(False)
	tbl.set_fontsize(6.5)
	plt.tight_layout()
	plt.show()

#gets best tm from data generated in evaluatenNumTopicsGraph functions
#returns the best model
def getBestModelFromEvalBestTopic(modelListCoherenceTuple):
	bestScore = (0, 0)
	i = 1

	for cv in modelListCoherenceTuple[1]:
		if(cv > bestScore[1]):
			bestScore = (i, cv)

		i += 1

	print('best models is: ' + str(bestScore[0]) + ' topics with ' + str(bestScore[1]) + ' coherence.')
	return modelListCoherenceTuple[0][bestScore[0] - 1]

def writeCoherenceToFile(filename, coherence):
	with open(filename, 'w+') as f:
		f.write('"NumTopics","Coherence"\n')
		i = 1
		for num in coherence:
			f.write('"' + str(i) + '"' + ',"' + str(num) + '"\n')
			i += 1
'''
















#create dictonary
#dictionary = corpora.Dictionary(tokenizedC)
#pprint(dictionary.token2id)

'''
pprint()
#remove holdouts
testCorpus = generateHoldOutList(corpusL, 100)
print('Holdouts Selected')
'''

#Format corpus for gensim
#theCorpus = MyCorpus.createCorpus(corpusL)

#write corpus to disk for later use
#corpora.MmCorpus.serialize('/tmp/corpus.mm', theCorpus)

#build tfidf model
#tfidf = models.TfidfModel(theCorpus)

#convert corpus from bag of words to tfidf
#corpus_tfidf = tfidf[theCorpus]





#####	Train LSI	#####

#train lsi/lsa model
#lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=25, power_iters=50, extra_samples=5)
#pprint(lsi.print_topics(num_topics=10, num_words=10))


#####	Train LDA 	#####

#lda = models.LdaModel(theCorpus, id2word=dictionary, num_topics=50, passes=10, update_every=2, iterations=300, alpha='auto')
#pprint(lda.print_topics(num_topics=50, num_words=15))


#####	Train HDP 	#####
'''
hdp = models.HdpModel(theCorpus, id2word=dictionary)
#hdp.print_topics(num_topics=20, num_words=10)
'''
#####	Output Topic Models 	#####
'''
keywordList = generateTopicKeywordList(lsi, 5, 10)
KeywordListToCSV(keywordList, 'lsi')
'''
'''
keywordList = generateTopicKeywordList(lda, 25, 10)
KeywordListToCSV(keywordList, 'lda')
'''
'''
keywordList = generateTopicKeywordList(hdp, 25, 10)
KeywordListToCSV(keywordList, 'hdp')
'''
#lsi.print_topics(-1)

#### VALIDATE MODEL LDA, perplexity, lower the better

'''
testSetChunk = []
for doc in testCorpus:
	testSetChunk.append(dictionary.doc2bow(MyCorpus.tokenizeDoc(doc)))
'''
'''
lm_list_cv = evaluateLDANumTopicsGraph(dictionary, theCorpus, tokenizedC, 201)
'''
'''
numTopicsToShow = 25


lm_list_cv = evaluateLDANumTopicsGraph(dictionary, corpus_tfidf, tokenizedC, 200)
bestModel = getBestModelFromEvalBestTopic(lm_list_cv)

numTopics = bestModel.num_topics
print('num topics of best model is ' + str(numTopics))
if(numTopics < numTopicsToShow):
	numTopicsToShow = numTopics

'''
#lda = models.LdaModel(corpus=theCorpus, num_topics=6, id2word=dictionary,  passes=12, update_every=2, iterations=250, alpha='auto', eta='auto')


#showBestTopics(lda, tokenizedC, dictionary, 2, 5, 10)
#writeCoherenceToFile('coherenceLDA.csv', lm_list_cv[1])



'''
print('Calculating perplexity on heledout documents:\n')

perplex = lda.log_perplexity(testSetChunk, total_docs=len(testCorpus))

print(lda.bound(testSetChunk))
'''
