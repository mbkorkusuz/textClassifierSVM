import nltk
import string
import pickle	# this is for saving and loading your trained classifiers.
from sklearn.svm import SVC
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, classification_report
import pandas as pd


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


def load_classifier(filename):	#filename should end with .pickle and type(filename)=string
	print("Loading Classifier")
	classifier_file = open(filename, "rb")
	classifier = pickle.load(classifier_file)
	classifier_file.close()
	return classifier

def test(classifier, test_set):
	
	y_pred = classifier.predict(test_set[0])
	y_true = test_set[1]  #labels

	classification_report(y_true=y_true, y_pred=y_pred)
	
	precision = precision_score(y_true, y_pred, average='micro')
	recall = recall_score(y_true, y_pred, average='micro')
	f1 = f1_score(y_true, y_pred, average = "micro")
	accuracy = accuracy_score(y_true, y_pred)

	print("Accuracy:",accuracy)
	print("F1-Score:",f1)
	print("Precision:",precision)
	print("Recall:",recall)

if __name__ == "__main__":
	#### testing ### 
	test_set = []
	test_set = create_megadoc() ## megadocument is created
	loadedVectorizer = pickle.load(open('countVectorizer.pickle', 'rb')) ## I am loading the vectorizer that I used in the training
	loadedClassifier = load_classifier("SVMClassifier.pickle") ## I am loading the classifier that I used in the training

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
