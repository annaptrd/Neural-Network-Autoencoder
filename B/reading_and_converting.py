import sys
from tensorflow import keras
from mlxtend.data import loadlocal_mnist



def readingArguements(argv):
	n=len(argv)
	for i in range(1,n):#diavasma twn parametrwn apo th grammh
		if(argv[i]=="-d"):
			training_set_file=argv[i+1]
		elif(argv[i]=="-dl"):
			training_labels_file=argv[i+1]
		elif(argv[i]=="-t"):
			test_set_file=argv[i+1]
		elif(argv[i]=="-tl"):
			test_labels_file=argv[i+1]
		elif(argv[i]=="-model"):
			model_file=argv[i+1]
			
	return training_set_file , training_labels_file , test_set_file , test_labels_file , model_file
	

def file_normalization(training_set_file , training_labels_file , test_set_file , test_labels_file):

	training_set, training_labels = loadlocal_mnist(images_path=training_set_file,labels_path=training_labels_file)#ta bytes twn arxeiwn apothikeyontai stoys pinakes training_set kai training_labels
	test_set, test_labels = loadlocal_mnist(images_path=test_set_file,labels_path=test_labels_file)#ta bytes twn arxeiwn apothikeyontai stoys pinakes test_set kai test_labels

	training_set = training_set/255#kanonikopoihsh twn pixels sto (0,1)
	test_set = test_set/255#kanonikopoihsh twn pixels sto (0,1)

	training_set = training_set.reshape(training_set.shape[0],28,28,1)#metattroph toy arxeioy eikonwn wste na einai ths morfhs pinaka 28*28*1
	test_set = test_set.reshape(test_set.shape[0],28,28,1)#metattroph toy arxeioy eikonwn wste na einai ths morfhs pinaka 28*28*1

	training_labels=keras.utils.to_categorical(training_labels, 10) #one-hot encode our data.
	test_labels=keras.utils.to_categorical(test_labels, 10) #one-hot encode our data.
	
	return training_set , training_labels , test_set , 	test_labels
	
