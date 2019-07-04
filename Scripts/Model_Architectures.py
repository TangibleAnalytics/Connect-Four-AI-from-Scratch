# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 17:59:50 2019

@author: Jake
"""

from keras import Input, Model
from keras.layers import Conv2D, Reshape, Flatten
from keras.layers import Dense, LeakyReLU, Dropout
from keras.layers import Concatenate, BatchNormalization

def Perception_Model_Definition(Board_Shape=(6,7), Action_Space=7):
    
    Board_State_Input = Input(shape=Board_Shape)
    x = Board_State_Input
    
    x = Reshape(target_shape=(Board_Shape[0], Board_Shape[1], 1))(x)
    for Conv_Layer in range(3):
        x = Conv2D(filters=100, kernel_size=(3,3), padding='same')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = Dropout(rate=0.1)(x)
        continue
    x = Flatten()(x)
    
    for Dense_Layer in range(3):
        x = Dense(units=100)(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = Dropout(rate=0.1)(x)
        continue
    
    #Incorporate action
    Action_Taken_Input = Input(shape=(Action_Space,))
    x = Concatenate()([x, Action_Taken_Input])
    
    x = Dense(units=1, activation='linear')(x)
    Model_Output = x
    
    Perception_Model = Model(inputs=[Board_State_Input, Action_Taken_Input], outputs=Model_Output)
    Perception_Model.compile(optimizer='adam', loss='mean_squared_error')
    return Perception_Model


def Policy_Model_Definition(Board_Shape=(6,7), Action_Space=7):
    
    Board_State_Input = Input(shape=Board_Shape)
    x = Board_State_Input
    
    x = Reshape(target_shape=(Board_Shape[0], Board_Shape[1], 1))(x)
    for Conv_Layer in range(2):
        x = Conv2D(filters=25, kernel_size=(3,3), padding='same')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = Dropout(rate=0.1)(x)
        continue
    x = Flatten()(x)
    
    for Dense_Layer in range(2):
        x = Dense(units=50)(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = Dropout(rate=0.1)(x)
        continue
    
    Action_Output = Dense(units=Action_Space, activation='softmax')(x)
    Policy_Model = Model(inputs=Board_State_Input, outputs=Action_Output)
    Policy_Model.compile(optimizer='adam', loss='categorical_crossentropy')
    return Policy_Model


'''
import numpy as np
X = np.zeros((1,6,7))
Y = np.array([[1,0,0,0,0,0,0]])

Policy_Model = Policy_Model_Definition()
#print(Policy_Model.summary())
Perception_Model = Perception_Model_Definition()
#print(Perception_Model.summary())
'''





