import nltk
import string
import pickle	# this is for saving and loading your trained classifiers.
from sklearn.svm import SVC
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
## I don't use all these imports

nltk.download("stopwords")
nltk.download('punkt_tab')
nltk.download('wordnet')

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
		lines[i] = lines[i].translate(str.maketrans('', '', string.punctuation)) ## punctuations removed
		lines[i] = word_tokenize(lines[i].lower()) ## tokenize and lowercase
		lines[i] = [lemmatizer.lemmatize(word) for word in lines[i]] #lemmatization
		lines[i] = [word for word in lines[i] if word not in sWords] # stopwords (the, will, can etc ) removed
		processed.append((lines[i],filename))

	return processed


def create_megadoc():
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

def train(classifier, training_set):	# classifier is either nltk.NaiveBayesClassifier or SklearnClassifier(SVC()). Example call: train(SklearnClassifier(SVC()), training_set)
	print("Training...")
	features=training_set[0]
	labels=training_set[1]
	classifier.fit(features,labels)
	save_classifier(classifier, "SVMClassifier.pickle") # save classifier

def save_classifier(classifier, filename):	#filename should end with .pickle and type(filename)=string
	print("Saving Classifier")
	with open(filename, "wb") as f:
		pickle.dump(classifier, f)
	return


if __name__ == "__main__":
	# You may add or delete global variables.
	training_set = []
	test_set = []

	training_set = create_megadoc() ## megadocument is created
	training_setCount = extract_features(training_set) ## feature extraction 

	classifierSVM = SVC()
	train(classifierSVM, training_setCount) # train
	print("Training Done")
