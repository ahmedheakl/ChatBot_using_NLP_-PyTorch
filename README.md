## ChatBot using NLP: Project Overview
* Built a Chatbot that sells coffee and tell jokes using Pytorch (loss ~ 0.003)
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

## Dataset
The data is in Json (Javascript Object Notation) formatting which can be imported as python dictionary: 
* The data consists of tags where each tag represents the type of sentences included under it
* Under each tag:
    * **Patterns:** Expected user sentences
    * **Responses:** Expected user responses

Here is a snippt of the data inside the *intents.json* file:
<p align="center">
<img width='300' height='400' src='https://github.com/ahmedheakl/ChatBot_using_NLP_-PyTorch/blob/main/data_snippt.PNG'>         
</p>


## Building Utils
* Built Tokenizer from nltk
* Built Stemmer from nltk (*PorterStemmer* used)
* Built ```bag_of_word``` to get the vector of the sentence to be used later in the model 

## Data Engineering
* Applied the utils built above to get all words of the dataset
* Get rid of punctuation words
* Create ```y_train``` list using the the index of the label tags

## Model, Training, and Perfomance
* The model is a two hidden-layer-linear model with ReLU activation in between. 
* Each hidden layer has 8 neurons.
* The input_size is the size of all the words in the dataset as each sentece is represented as vector of zeros and ones with the length of all the words in the dataset
* The num_classes is the number of tags out there. 

**Here is a summary of the model:**
<p align="center">
<img width='400' height='300' src='https://github.com/ahmedheakl/ChatBot_using_NLP_-PyTorch/blob/main/Model_photo.jpg'>         
</p>

* Trained the model using Adam optimizer with a ```learning_rate=0.001``` for ```num_epochs=1000```
* The model performed very well with a very small CrossEntropyLoss of ~ 0.003.

## Bot Implementation
* Prompted the user with an entry sentence 
* Tokenize and vectorized the input sentence from the user
* Reshaped and coverted the vector into a tensor
* Predicting which tag the sentence belongs to:
     * if the ```probability``` of the prediction ```> 0.7``` we prompt the user with a randomly chosen reponse from this tag
     * if not we prompt the use with *"I don't understand"*
* The user can quit the chat by typing *quit*

Here is an example: 
<p align="center">
<img width='500' height='400' src='https://github.com/ahmedheakl/ChatBot_using_NLP_-PyTorch/blob/main/example.PNG'>         
</p>



