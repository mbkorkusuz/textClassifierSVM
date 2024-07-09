import nltk
import sklearn
import string
import pickle	# this is for saving and loading trained classifiers.

from sklearn.svm import SVC
from nltk.classify.scikitlearn import SklearnClassifier

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn import model_selection, svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
import random
import pandas as pd
import sklearn_crfsuite


def preprocess(filename):	
	print("Preprocess...")
	filepath = filename					
	file = open(filepath, 'r')			
	lines = file.read().splitlines()	
	file.close()

	genres = ["philosophy", "sports", "mystery", "religion", "science", "romance", "horror", "science-fiction"] 

	for i in genres:
		if i in filename:
			filename = genres.index(i) + 1
			break
	
	processed = []		
					
	lemmatizer = nltk.stem.WordNetLemmatizer()
	sWords = stopwords.words("english")
	for i in range(1, len(lines), 2):
		lines[i] = lines[i].translate(str.maketrans('', '', string.punctuation)) # punctuations removed
		lines[i] = word_tokenize(lines[i].lower()) # tokenize and lowercase
		lines[i] = [lemmatizer.lemmatize(word) for word in lines[i]] #lemmatization
		lines[i] = [word for word in lines[i] if word not in sWords] # stopwords (the, will, can etc ) removed
		processed.append((lines[i],filename))

	return processed


def create_training_megadoc():
	print("Converting Training Megadoc...")
	training_documents = ["data/train/philosophy_train.txt","data/train/sports_train.txt","data/train/mystery_train.txt","data/train/religion_train.txt","data/train/science_train.txt","data/train/romance_train.txt","data/train/horror_train.txt","data/train/science-fiction_train.txt"]
	training_megadoc = []
	for filename in training_documents:
		training_megadoc += preprocess(filename)

	stories = []
	labels = []

	for i in training_megadoc:
		stories.append(i[0])
		labels.append(i[1])

	pandaDataFrame = pd.DataFrame()
	pandaDataFrame['story'] = stories
	pandaDataFrame['labels'] = labels

	pandaDataFrame['story'] = pandaDataFrame['story'].map(' '.join)

	return pandaDataFrame

def create_test_megadoc():
	print("Converting Testing Megadoc...")
	test_documents = ["data/test/philosophy_test.txt", "data/test/sports_test.txt", "data/test/mystery_test.txt", "data/test/religion_test.txt", "data/test/science_test.txt", "data/test/romance_test.txt", "data/test/horror_test.txt", "data/test/science-fiction_test.txt"]	
	test_megadoc = []
	
	for filename in test_documents:
		test_megadoc += preprocess(filename)

	stories = []
	labels = []

	for i in test_megadoc:
		stories.append(i[0])
		labels.append(i[1])

	pandaDataFrame = pd.DataFrame()
	pandaDataFrame['story'] = stories
	pandaDataFrame['labels'] = labels

	pandaDataFrame['story'] = pandaDataFrame['story'].map(' '.join)

	return pandaDataFrame
		


def extract_features(megadoc):		
	
	print("Extracting Features...")
	text = megadoc['story']
	labels = megadoc['labels']

	countVector = CountVectorizer()
	countVector.fit(text)
	text = countVector.transform(text)

	# Save the vectorizer
	vecFile = 'countVectorizer.pickle'
	pickle.dump(countVector, open(vecFile, 'wb'))

	newLabels = []

	for i in labels:
		newLabels.append(i)
	text = text.toarray()
	featuredText = [text, newLabels]

	return featuredText


def train(classifier, training_set, type):
	print("Training...")
	features=training_set[0]
	labels=training_set[1]
	classifier.fit(features,labels)
	if type == 1:
		save_classifier(classifier, "SVMClassifier.pickle") # save classifier

def test(classifier, test_set):
	
	y_pred = classifier.classify(test_set[0])
	y_true = test_set[1]  #labels
	
	precision = sklearn.metrics.precision_score(y_true, y_pred, average='micro')
	recall = sklearn.metrics.recall_score(y_true, y_pred, average='micro')
	f1 = 2*(precision*recall)/(precision+recall)
	accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)

	print("Accuracy:",accuracy)
	print("F1:",f1)
	print("Precision:",precision)
	print("Recall:",recall)
	

def save_classifier(classifier, filename):
	print("Saving Classifier")
	with open(filename, "wb") as f:
		pickle.dump(classifier, f)
	return

def load_classifier(filename):
	print("Loading Classifier")
	classifier_file = open(filename, "rb")
	classifier = pickle.load(classifier_file)
	classifier_file.close()
	return classifier


if __name__ == "__main__":

	training_set = []
	test_set = []

	training_set = create_training_megadoc() # megadocument is created
	training_setCount = extract_features(training_set) # feature extraction 

	classifierSVM = SVC()
	train(classifierSVM, training_setCount, 1) # train


	# training is done now let's test the created classifier file

	#### testing 
	test_set = create_test_megadoc() ## megadocument is created
	loadedVectorizer = pickle.load(open('countVectorizer.pickle', 'rb')) # loading the vectorizer that I used in the training
	loadedClassifier = load_classifier("SVMClassifier.pickle") # loading the classifier that I used in the training

	# feature extraction 
	text = test_set['story'] 
	labels = test_set['labels']
	text = loadedVectorizer.transform(text)
	text = text.toarray()
	newLabels = []
	for i in labels:
		newLabels.append(i)
	featuredTestSet = [text, newLabels]

	test(loadedClassifier, featuredTestSet) ## testing

	print("Done...")
	