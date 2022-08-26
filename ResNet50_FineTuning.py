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
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array



np.random.seed(0)
#dataframe con label a predecir (cobertura,target,etc) y filename de los tiles 
dataTrain=pd.read_csv("../dataTrain_label.csv")
dataTest=pd.read_csv("../dataTest_label.csv")
dataVal=pd.read_csv("../dataVal_label.csv")

datagen = preprocessing.image.ImageDataGenerator()
#data augmentation and preprocessing
#preprocessing_function especial para ResNet50
datagen = preprocessing.image.ImageDataGenerator(featurewise_center=True,rotation_range=40,horizontal_flip=True,vertical_flip=True)
datagen.mean = [123.68, 116.779, 103.939]
#x_col=filename de los tiles 
#y_col= target a predecir 
#class_mode="raw" , para regresión 
train_it = datagen.flow_from_dataframe(dataframe=dataTrain,directory="../dataset_rgb/",x_col="filename",y_col="label",batch_size=64,class_mode="raw",target_size=(224,224),shuffle=True)
test_it= datagen.flow_from_dataframe(dataframe=dataTest,directory="../test/",x_col="filename",y_col="label",batch_size=64,class_mode="raw",target_size=(224,224),shuffle=True)

#prediction para evaluar desempeño del modelo. 
def prediction(modelo,dataset,directorio):

    predict= np.zeros(len(dataset))
    for i in range(len(dataset)):
        img = load_img(directorio+dataset.iloc[i,0], target_size=(224, 224))
        img = img_to_array(img)
        img = img.reshape(1,224,224, 3)
        img = img.astype('float32')
        img = img - [123.68, 116.779, 103.939]
        result = modelo.predict(img)
        predict[i]= result[0][0]
    return predict 
#evaluacion del modelo con MSE,R-cuadrado,RMSE y MAE. 
def metrics(real,predict):
    mse=mean_squared_error(real,predict)
    r2=r2_score(real,predict)
    rmse=np.sqrt(mean_squared_error(real,predict))
    mae=mean_absolute_error(real,predict)
    return rmse,r2,mae,mse


start_time = time.time()
opt = tf.keras.optimizers.RMSprop(learning_rate=0.0001)#para definir el optimizers y learning_rate
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience= 10,mode='min')#early_stopping para detener el entrenamiento y evitar el overfitting
#patience= 10 para detener el entrenamiento 10 épocas después del mejor resultado. 
#monitor='val_loss' monitorear el modelo mediante esta métrica 
#mode='min' buscar el mínimo error 
checkpoint=callbacks.ModelCheckpoint("BestModelResNet.hdf5",monitor='val_loss',save_best_only=True,mode='min')#Save al mejor modelo. 

model = ResNet50(include_top=False,input_shape=(224,224, 3))

#fine tuning 
flat1 = layers.Flatten()(model.layers[-1].output)
class1 = layers.Dense(128, activation='relu')(flat1)# se incluye una capa de 128 neuronas, con función de activación "relu"
output = layers.Dense(1, activation='sigmoid')(class1)# capa de salida con sigmoid para medir cobertura. (Puede ser función "linear")

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
bestModel = load_model("BestModelResNet.hdf5")
 
prediction_train=prediction(bestModel,dataTrain,"../dataset_rgb/")#evaluacion train
prediction_test=prediction(bestModel,dataTest,"../test/") # evaluacion test 
prediction_val=prediction(bestModel,dataVal,"../validation/")#evaluacion val
   
    
desem_train = metrics(dataTrain.iloc[:,1],prediction_train)#metricas 
desem_test= metrics(dataTest.iloc[:,1],prediction_test)
desem_val= metrics(dataVal.iloc[:,1],prediction_val)
    
print("Desempeño train : ",desem_train)
print("Desempeño test : ",desem_test)
print("Desempeño validacion : ",desem_val)
    
#Para graficar real versus predicción. 
#(ggplot(aes(x = dataVal.iloc[:,1],y = prediction_val)) + 
  #scale_y_continuous(limits=(0,1))+
  #geom_point(alpha=0.5, color='blue')+
  #labs(x="Label",y="Predict")+
  #geom_abline(intercept=0,color='black')+
  #annotate("text",label="R 0.74",color="red",x=0.5,y=0.83)+
  #annotate("text",label="RMSE 0.09",color="red",x=0.5,y=0.90))

