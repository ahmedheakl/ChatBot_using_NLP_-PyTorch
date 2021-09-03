## ChatBot using NLP: Project Overview
* Built a Chatbot that sells coffee and tell jokes using Pytorch (acc ~ 0.89)
* Tokenized, stemmed, converted to bag of words the data to be used later by the model.
* Built a 2 layer model using pytorch and trained it using Adam optimizer
* Implemented the chatbot for to output English senetence based on the model prediction

## Resources
**Data retrived from:** https://github.com/python-engineer/pytorch-chatbot/blob/master/intents.json

**DataLoader Tutorial:** https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

## Code
**Python Version:** 3.9.6
**Packages:** Numpy, PyTorch, NLTK, Json
**For training:** I used my *RTX 2060* to train (done in ~ 30 secs) 

## Building Utils
* Built Tokenizer from nltk
* Built Stemmer from nltk (*PorterStemmer* used)
* Built ```bag_of_word``` to get the vector of the sentence to be used later in the model 

## Data Engineering
* Applied the utils built above to get all words of the dataset
* Get rid of punctuation words
* Create ```y_train``` list using the the index of the label tags

## Model

  <img width='350' height='300' src='https://github.com/ahmedheakl/ChatBot_using_NLP_-PyTorch/blob/main/Model_photo.jpg'>          

