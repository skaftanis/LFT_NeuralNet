import pickle
import sys
import numpy as np

"""sigmoid function and its derivitive"""
def nonlin(x,deriv=False):

	if (deriv==True):
		return x*(1-x)

	return 1/(1+np.exp(-x))  

"""Compute softmax values for each sets of scores in x."""
def softmax(x):
	return np.exp(x) / np.sum(np.exp(x), axis=0)


#loads a picle file with the weights of the neural net 
with open('weights.pickle') as f:  # Python 3: open(..., 'rb')
    syn0, syn1 = pickle.load(f)



inputToPredictStr =  sys.argv[1]
 
#tokenize a list-like string 
day = int( inputToPredictStr.split(",")[0].replace('[',"").replace(']',"") )
time = int( inputToPredictStr.split(",")[1].replace('[',"").replace(']',"") )
store_id = int( inputToPredictStr.split(",")[2].replace('[',"").replace(']',"") )


realPredictionInput = [ day, time, store_id]

realPredictionInput = np.array(realPredictionInput)

#passing data thought the neural network 
l1new = nonlin(np.dot(realPredictionInput,syn0)) #layer1 
l2new = nonlin(np.dot(l1new,syn1)) #layer2

#aply the softmax for normalization
l2new=softmax(l2new)

prediction = l2new.argmax()

print "{\"results\":[{\"pred\":\"" ,  prediction, "\"", "}]}"
