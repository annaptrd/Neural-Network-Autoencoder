from keras.models import load_model,Model
import tensorflow as tf
from keras.layers import Input,Flatten,Activation,Dense,Dropout

def build_model(model_file,fc_nodes):
	model=load_model(model_file)
	#model.layers[0].get_weights()
	classification=tf.keras.Sequential()
	flag=0
	for layer in model.layers:	#diavazetai to modelo kai krateitai mono to encoder kommati
		if(layer.get_output_at(0).get_shape().as_list()[1]==7):
			flag=1
		if(flag==1 and layer.get_output_at(0).get_shape().as_list()[1]!=7):
			break
		
		classification.add(layer)								
													
	for layer in classification.layers:		#ta layers toy encoder ginontai non trainable
		layer.trainable=False

	classification.add(Flatten())
	classification.add(Dense(fc_nodes,activation='relu'))
	classification.add(Dropout(0.5, input_shape=(784,)))
	classification.add(Dense(10,activation='softmax'))
	#classification.summary()
	
	return classification
