from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import random

def print_classification_reports(test_Accuracies , test_Losses , hidden_units , epochs , batches , learning_rates , test_set , test_labels , predictions ):
	for i in range(0,len(test_Accuracies)):		#ektypwsh olwn twn classification reports gia ola ta peiramata
		print("For the experiment with", hidden_units[i] ," Hidden Units, ",epochs[i]," Epochs, ",batches[i]," Batch Size, ",learning_rates[i]," Learning Rate:\n")
		print(f'Test loss: {test_Losses[i]} \nTest accuracy: {test_Accuracies[i]}\n')
		correctlabels=int(len(test_set)*test_Accuracies[i])
		print("Found ",correctlabels," correct labels")
		wronglabels=len(test_set)-correctlabels
		print("Found ",wronglabels," incorrect labels\n")
		target_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7', 'Class 8', 'Class 9']
		print(classification_report(test_labels, predictions[i], target_names=target_names))
	
	

def display_graph_option(hidden_units , epochs , batches , learning_rates , train_Accuracies , train_Losses , test_Accuracies , test_Losses ):
	print("Which graphs you want to be displayed?")
	print("1)The ones with the Hidden Units")
	print("2)The ones with the Epochs")
	print("3)The ones with the Batch Size")
	print("4)The ones with the Learning Rate")
	graphoption=input("Enter 1, 2, 3 or 4: ")		#epilogh emfanishs diagrammatwn
	if(graphoption=="1"):
		printGraphs(hidden_units,train_Accuracies,train_Losses,test_Accuracies,test_Losses,"hidden units")		#epilogh fc units
	elif(graphoption=="2"):
		printGraphs(epochs,train_Accuracies,train_Losses,test_Accuracies,test_Losses,"Epoch")		#epilogh epochs
	elif(graphoption=="3"):
		printGraphs(batches,train_Accuracies,train_Losses,test_Accuracies,test_Losses,"Batch Size")		#epilogh batch size
	elif(graphoption=="4"):
		printGraphs(learning_rates,train_Accuracies,train_Losses,test_Accuracies,test_Losses,"Learning Rate")	#epilogh learning rate
	


def printGraphs(param_array,train_Accuracies,train_Losses,test_Accuracies,test_Losses,param):
	plt.plot(param_array,train_Losses,'-',label='training_loss')
	plt.plot(param_array,test_Losses,'-',label='test_loss')
	plt.legend()
	plt.title('Training and testing loss')
	plt.xlabel(param)
	plt.show()
	
	plt.plot(param_array,train_Accuracies,'-',label='training_accuracy')
	plt.plot(param_array,test_Accuracies,'-',label='test_accuracy')
	plt.legend()
	plt.title('Training and testing accuracy')
	plt.xlabel(param)
	plt.show()
	
	
def image_classification(hidden_units , epochs , batches , learning_rates , classifications , test_set , test_labels):
	print("Please enter the hyperparameters you want to use for the classification of the images: \n")
	num_of_chosen_experiment=-1
	while num_of_chosen_experiment==-1:
		chosen_units=int( input("Enter the Hidden Units of the fully connected layer: ") )
		chosen_epochs=int( input("Enter the number of Epochs: ") )
		chosen_batches=int( input("Enter the Batch Size: ") )
		chosen_rate=float( input("Enter the Learning Rate: ") )
		for i in range(0,len(hidden_units)):	#image classification gia sygkekrimeno peirama
			if(hidden_units[i]==chosen_units and epochs[i]==chosen_epochs and batches[i]==chosen_batches and learning_rates[i]==chosen_rate):
				num_of_chosen_experiment=i
				break
		if(num_of_chosen_experiment==-1):
			print("There is no experiment with these hyperparameters,Please enter an existing experiment!")
		
			
	print("The experiment with ",hidden_units[num_of_chosen_experiment], " hidden units , ",epochs[num_of_chosen_experiment]," Epochs , ",batches[num_of_chosen_experiment]," batch size and  ",learning_rates[num_of_chosen_experiment]," learning rate is chosen")
	columns = 4
	rows = 4
	print("Which images you want to be displayed?")
	option=input("Press 1)if you want to display the right predicted images or 2)if you want to display the wrong predicted images:")
	print("A sample of ",columns*rows," pictures is gonna show up for this experiment")
	fig=plt.figure(figsize=(8, 8))
	num_of_images=0
	chosen_indexes=[]
	while(num_of_images<columns*rows):
		image_index=random.randint(0, len(test_set)-1 )
		if(image_index not in chosen_indexes ):
			chosen_indexes.append(image_index)
			pred = classifications[num_of_chosen_experiment].predict(test_set[image_index].reshape(1, 28, 28, 1))
			true_class=0
			for i in range(0,len(test_labels[image_index]) ):
				if(test_labels[image_index][i].item()==1):
					#print("FTASAME:",test_labels[image_index])
					true_class=i
					#print("i=",true_class)
					break
			
			if(option=="2"):
				if(true_class==pred.argmax()):	#psaxnoume tis lathos eikones
					continue
					
			fig.add_subplot(rows, columns, num_of_images+1)
			plt.imshow(test_set[image_index].reshape(28, 28),cmap=plt.get_cmap('gray'))
			num_of_images+=1
			plt.title('Predicted %d'% (pred.argmax()) + ',Class %d' %(true_class),fontweight ="bold",fontsize=11)
			
				
	plt.subplots_adjust(left=0.125,bottom=0.1,right=0.9,top=0.9,wspace=0.4,hspace=0.35)
	plt.show()
	
	
	
	
	
	
	
