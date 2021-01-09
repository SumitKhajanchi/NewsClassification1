from django.shortcuts import render
from django.http import HttpResponse
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pickle

modulePath = os.path.dirname(__file__) #get current directory

filePath = os.path.join(modulePath, 'tfidftransformer.tfidf') 
with open(filePath, 'rb') as f:
	tfidfvec = pickle.load(f)

filePath = os.path.join(modulePath, 'finalized_model.model')
with open(filePath,  'rb') as f:
	model = pickle.load(f)

filePath = os.path.join(modulePath, 'id_dict.pickle')
with open(filePath, 'rb') as f:
	id_dict = pickle.load(f)

def text_lowercase(text):
	return text.lower()

def classify(request):
	# Get the text from client side, if we dont get any text then set the text = default
	djtext = request.GET.get('text', 'default')

	# If text != default
	if djtext != "default":
		text = [text_lowercase(djtext)]
		features = tfidfvec.transform(text).toarray()
		predicted = model.predict(features)
		predicted = id_dict[predicted[0]]

	if djtext == "default":
		predicted = "No text provided please try again"

	params = {'Category': predicted}
	return render(request, 'result.html', params)

def home(request):
	return render(request, 'index.html')