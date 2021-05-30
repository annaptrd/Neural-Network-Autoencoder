import sys
import numpy as np
from tensorflow.keras import layers,losses,optimizers,metrics
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import warnings
from classification_results import printGraphs , print_classification_reports , display_graph_option , image_classification
from reading_and_converting import readingArguements , file_normalization
from classifier_building import build_model

warnings.filterwarnings('ignore')

np.set_printoptions(threshold=sys.maxsize)


training_set_file , training_labels_file , test_set_file , test_labels_file , model_file = readingArguements(sys.argv) 	#reading the arguements

training_set , training_labels , test_set , test_labels=  file_normalization(training_set_file , training_labels_file , test_set_file , test_labels_file)		#normalize the files


hidden_units=[]		#pinakas poy fylaei ta units twn peiramatwn mas
epochs=[]			#pinakas poy fylaei ta epochs twn peiramatwn mas
batches=[]			#pinakas poy fylaei ta batches twn peiramatwn mas
learning_rates=[]	#pinakas poy fylaei ta lr twn peiramatwn mas
classifications=[]	
predictions=[]		
test_Accuracies=[]	#pinakas poy fylaei to evaluate accuracy tou test gia kathe experiment
test_Losses=[]		#pinakas poy fylaei to evaluate loss tou test gia kathe experiment
train_Accuracies=[]	#pinakas poy fylaei to evaluate accuracy tou train gia kathe experiment
train_Losses=[]		#pinakas poy fylaei to evaluate loss tou train gia kathe experiment

while True:										
	fc_nodes=int(input("Enter the number of hidden units of the fully connected layer: "))
	hidden_units.append(fc_nodes)
	numberOfEpochs=int(input("Enter the number of epochs: "))
	epochs.append(numberOfEpochs)
	batchSize=int(input("Enter the batch size: "))
	batches.append(batchSize)
	learningRate=float(input("Enter the learning rate: "))
	learning_rates.append(learningRate)

	print(hidden_units)
	print(epochs)
	print(batches)
	print(learning_rates)
	
	classification=build_model(model_file,fc_nodes)	#construction of the classification_model

	classification.compile(loss=losses.CategoricalCrossentropy(), optimizer=RMSprop(learning_rate = learningRate),metrics=[metrics.CategoricalAccuracy('accuracy')])
	#classification.compile(loss=losses.CategoricalCrossentropy(), optimizer='adam',metrics=[metrics.CategoricalAccuracy('accuracy')])
	FC_train = classification.fit(training_set, training_labels , validation_split=0.1 , batch_size=batchSize,epochs=numberOfEpochs, verbose=1)
	print("The train of the fully connected layer with number of nodes ",fc_nodes," is finished")
	print("It's time to train the whole model now")

	for layer in classification.layers:		#ta layers toy encoder ginontai kai ayta trainable
		layer.trainable=True
							
	#classification.summary()
	
	classification.compile(loss=losses.CategoricalCrossentropy(), optimizer=RMSprop(learning_rate = learningRate),metrics=[metrics.CategoricalAccuracy('accuracy')])
	#classification.compile(loss=losses.CategoricalCrossentropy(), optimizer='adam',metrics=[metrics.CategoricalAccuracy('accuracy')])
	history=classification.fit(training_set, training_labels,validation_split=0.1, epochs=numberOfEpochs , batch_size=batchSize , verbose=1)
	classifications.append(classification)		#apothikeyetai to classification
	
	
	"""sum_losses=FC_train.history['loss']+history.history['loss']		#pinakas gia ta loss prwths kai deyterhs fashs
	valid_sum_losses=FC_train.history['val_loss']+history.history['val_loss']	#pinakas gia ta valid loss prwths kai deyterhs fashs
	epocharray=range(0,numberOfEpochs*2)
	plt.plot(epocharray,sum_losses,label='loss')	#anaparastisi toy loss se synarthsh me ta epochs
	plt.plot(epocharray,valid_sum_losses,label='val_loss')		#anaparastisi toy val_loss se synarthsh me ta epochs
	plt.legend()
	plt.title('Training and validation loss')
	plt.xlabel('Epochs')
	plt.show()
	
	sum_accuracies=FC_train.history['accuracy']+history.history['accuracy']		#pinakas gia ta accuracy prwths kai deyterhs fashs
	valid_sum_accuracies=FC_train.history['val_accuracy']+history.history['val_accuracy']	#pinakas gia ta valid accuracy prwths kai deyterhs fashs
	epocharray=range(0,numberOfEpochs*2)
	plt.plot(epocharray,sum_accuracies,'-',label='accuracy')
	plt.plot(epocharray,valid_sum_accuracies,'-',label='val_accuracy')
	plt.legend()
	plt.title('Training and validation accuracy')
	plt.xlabel('Epochs')
	plt.show()"""
	
	
	
	y_pred = classification.predict(test_set)	#predictions twn eikonwn
	thresh = 0.5
	y_pred_normalization = np.array([[1 if i > thresh else 0 for i in j] for j in y_pred])		#metatroph se hard classes wste o pinakas na exei 0 h 1
	predictions.append(y_pred_normalization)
	
	train_score = classification.evaluate(training_set, training_labels, verbose=1)		#evaluation toy training set
	train_Accuracies.append(train_score[1])
	train_Losses.append(train_score[0])
	print("score for train=",train_score[0],"     ",train_score[1])
	
	test_score = classification.evaluate(test_set, test_labels, verbose=1)		#evaluation toy test set
	print("score for test=",test_score[0],"     ",test_score[1])
	test_Accuracies.append(test_score[1])
	test_Losses.append(test_score[0])
	
	
	option="0"
	while True:
		print("You can choose to: \n1) Repeat the experiment with different hyperparameters\n")
		print("2) Display classification reports and error graphs based on the hyperparameteres of the executed experiments\n")
		print("3) Classify the images\n")
		print("4)Terminate the program\n")
		option = input("Enter 1, 2, 3 or 4: ")

		while (option != "1") and (option != "2") and (option != "3") and (option!="4"):
			print("Please enter 1,2,3 or 4")
			option = user_input()
		if option == "1":
			break	#ksekiname neo peirama
			
		elif option == "2":
			print_classification_reports(test_Accuracies , test_Losses , hidden_units , epochs , batches , learning_rates , test_set , test_labels , predictions ) 	#emfanish classification reports
			display_graph_option( hidden_units , epochs , batches , learning_rates , train_Accuracies , train_Losses , test_Accuracies , test_Losses )
			
			continue_option="0"
			while continue_option!="2":
				print("What do you want to do now?")
				print("1)Display another graph")
				print("2)Continue")
				continue_option=input("Enter 1 or 2: ")
				while( (continue_option!="1") and (continue_option!="2") ):
					print("Please enter 1 or 2")
					continue_option = user_input()
				if continue_option=="1":
					display_graph_option( hidden_units , epochs , batches , learning_rates , train_Accuracies , train_Losses , test_Accuracies , test_Losses ) 	#emfanise diagramma kat'epiloghn
				elif(continue_option=="2"):
					break
					
		elif option == "3":
			image_classification(hidden_units , epochs , batches , learning_rates , classifications , test_set , test_labels)
		elif option=="4":
			break

	if(option=="4"):	#termatismos programmatos
		break


