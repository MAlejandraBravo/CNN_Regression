import pandas as pd
import numpy as np
import math
import time
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from tensorflow.keras import *
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input


np.random.seed(0)#inicilizar semilla de aletoriedad para el experimento
dataTrain=pd.read_csv("../dataTrain_label.csv")
dataTest=pd.read_csv("../dataTest_label.csv")
dataVal=pd.read_csv("../dataVal_label.csv")


datagen = preprocessing.image.ImageDataGenerator()
datagen = preprocessing.image.ImageDataGenerator(featurewise_center=True,rotation_range=40,horizontal_flip=True,vertical_flip=True,preprocessing_function=preprocess_input)
train_it = datagen.flow_from_dataframe(dataframe=dataTrain,directory="../dataset_rgb/",x_col="filename",y_col="label",batch_size=64,class_mode="raw",target_size=(299,299),shuffle=True)
test_it= datagen.flow_from_dataframe(dataframe=dataTest,directory="../test/",x_col="filename",y_col="label",batch_size=64,class_mode="raw",target_size=(299,299),shuffle=True)


def prediction(modelo,dataset,directorio):

    predict= np.zeros(len(dataset))
    for i in range(len(dataset)):
        img = load_img(directorio+dataset.iloc[i,0], target_size=(299,299))
        img = img_to_array(img)
        img = img.reshape(1,299,299, 3)
        img = preprocess_input(img)
        result = modelo.predict(img)
        predict[i]= result[0][0]
    return predict 

def metrics(real,predict):
    mse=mean_squared_error(real,predict)
    r2=r2_score(real,predict)
    rmse=np.sqrt(mean_squared_error(real,predict))
    mae=mean_absolute_error(real,predict)
    return rmse,r2,mae,mse


start_time = time.time()
opt = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience= 10,mode='min')
checkpoint=callbacks.ModelCheckpoint("BestModelInceptionV3.hdf5",monitor='val_loss',save_best_only=True,mode='min')

model = InceptionV3(include_top=False,input_shape=(299, 299, 3))

flat1 = layers.Flatten()(model.layers[-1].output)
class1 = layers.Dense(128, activation='relu')(flat1)
output = layers.Dense(1, activation='sigmoid')(class1)

# define new model
model = Model(inputs=model.inputs, outputs=output)
# compile model
model.compile(optimizer=opt, loss='mse', metrics=["mean_squared_error","mean_absolute_error"])
#fit model
model.fit(train_it, steps_per_epoch=len(train_it),validation_data=test_it, validation_steps=len(test_it), epochs=100,verbose=1,callbacks = [checkpoint,early_stopping])

seg=time.time() - start_time
hora=seg/3600
print("Tiempo de trabajo:",hora)


## evalución de modelo 
bestModel = load_model("BestModelInceptionV3.hdf5")
 
prediction_train=prediction(bestModel,dataTrain,"../dataset_rgb/")#evaluacion train
prediction_test=prediction(bestModel,dataTest,"../test/") # evaluacion test 
prediction_val=prediction(bestModel,dataVal,"../validation/")#evaluacion val
   
    
desem_train = metrics(dataTrain.iloc[:,1],prediction_train)#metricas 
desem_test= metrics(dataTest.iloc[:,1],prediction_test)
desem_val= metrics(dataVal.iloc[:,1],prediction_val)
    
print("Desempeño train : ",desem_train)
print("Desempeño test : ",desem_test)
print("Desempeño validacion : ",desem_val)
    

#(ggplot(aes(x = dataVal.iloc[:,1],y = prediction_val)) + 
  #scale_y_continuous(limits=(0,1))+
  #geom_point(alpha=0.5, color='blue')+
  #labs(x="Label",y="Predict")+
  #geom_abline(intercept=0,color='black')+
  #annotate("text",label="R 0.74",color="red",x=0.5,y=0.83)+
  #annotate("text",label="RMSE 0.09",color="red",x=0.5,y=0.90))

