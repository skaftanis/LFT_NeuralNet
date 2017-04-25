import urllib2
import simplejson
import calendar
import numpy as np

#load ALL_INPUTS db
response = urllib2.urlopen("http://showmeyourcode.org/get_all_inputs.php")
data = simplejson.load(response)

#load sizes
sizes = urllib2.urlopen("http://showmeyourcode.org/getSizes.php")
sizesData = simplejson.load(sizes)

sizeIds=[]
sizeValues=[]
for i in range(len(sizesData['results'])):
	sizeIds.append(sizesData['results'][i]['id'])
	sizeValues.append(sizesData['results'][i]['size'])


#for X
allInputs = []
allTimes = []
allIds = []


lengthOfList = len(data['results'])

#set the values
for i in range(lengthOfList):
	allIds.append(data['results'][i]['stores_id'])
	allInputs.append(data['results'][i]['input'])
	allTimes.append(data['results'][i]['time_entered'])

#manipulate times extract day of the week and time
#[day of week, time, store id]

allDaysOfWeek = []
allHours = []
for i in range(lengthOfList):
	allDaysOfWeek.append(calendar.weekday(int(allTimes[i][0:4]), int(allTimes[i][5:7]), int(allTimes[i][8:10])))
	allHours.append(int(allTimes[i][11:13]))

#free some space
allTimes = []

#input to neural net [allDaysOfWeek[i], allHours[i], allIds[i]] ..done
X = []
for i in range(lengthOfList):
	X.append( [  float(allDaysOfWeek[i]),float(allHours[i]),float(allIds[i])] )

#create y value (real value) as one hot vector [empty, almost empty, half, almost full, full]
y = []
for i in range(lengthOfList):
	size = sizeValues[sizeIds.index(allIds[i])]
	size = float(size)
	allInputs[i] = float(allInputs[i])
	if allInputs[i] >= (size - 0.05*size):
		y.append([1,0,0,0,0]) #empty
	elif allInputs[i] >= (size - 0.25*size):
		y.append([0,1,0,0,0]) #almost empth
	elif allInputs[i] >= (size - 0.5*size):
		y.append([0,0,1,0,0]) #half 
	elif allInputs[i] >= (size - 0.85*size):
		y.append([0,0,0,1,0]) #almost full
	else:
		y.append([0,0,0,0,1]) #full


X = np.array(X)
y = np.array(y)

#neural network staff 
np.random.seed(1)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def nonlin(x,deriv=False):
	if (deriv==True):
		return x*(1-x)

	return 1/(1+np.exp(-x))  

#synapses
syn0 = 2*np.random.random((3,10)) - 1
#print("syn0")
#print(syn0)
#x=np.ones(4)
#syn0[2,:]=x
syn1 = 2*np.random.random((10,5)) - 1 
#print("syn1")
#print(syn1)

#training step
for j in range(60000):

	l0 = X

	l1 = nonlin(np.dot(l0, syn0))
	l2 = nonlin(np.dot(l1,syn1))

	l2_error = y - l2 
	
	if (j % 10000) == 0: 
		print ("Error:" + str(np.mean(np.abs(l2_error))) )

	l2_delta = l2_error*nonlin(l2,deriv=True)

	l1_error = np.dot(l2_delta,syn1.T) # np.dot(l2_delta*syn1.T)
	l1_delta = l1_error * nonlin(l1,deriv=True)

	#update weights (gradient descent)
	syn1 += np.dot(l1.T,l2_delta)

	syn0 += np.dot(l0.T,l1_delta)



import pickle


#Saving weights in a file
with open('weights.pickle', 'w') as f:  
	pickle.dump([syn0, syn1], f)

print "done"
