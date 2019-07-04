# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 20:15:03 2019

@author: Jake
"""

import pickle
import Connect_Four as CF
import numpy as np
import time
import pandas as pd


#Set random seed based upon real-time clock
np.random.seed(round(time.time()))

Project_Base_Path = 'C:/Users/Jake/Documents/Actual/Data Science/Projects/Artificial Intelligence/Specialized/Proof House/Connect Four'

#Player One
Player_One_Path = Project_Base_Path + '/Data/Player One/Versions/Chad_0.p'
Player_One = pickle.load(open(Player_One_Path, 'rb'))

#Player Two
Player_Two_Path = Project_Base_Path + '/Data/Player Two/Versions/JT_0.p'
Player_Two = pickle.load(open(Player_Two_Path, 'rb'))

#Initialize Game Operator
Game_Meta_Data_Path = Project_Base_Path + '/Data/Games Meta Data/Game_Operator_Observations.csv'


Player_One.Learn(P_BatchSize=500, P_Epochs=1, C_BatchSize=500, C_Epochs=1)

'''
for i in range(200):

    Rounds = 1000
    for Round in range(Rounds):
        Operator = CF.Game_Operator(Player_One, Player_Two, Game_Meta_Data_Path, Game_ID=None)
        Operator.Execute_Round()
        if (Round % 100) == 0:
            print('Round: ' + str(Round))
        continue
    
    #Rewards_Bandwidth = 3
    Player_One.Learn(P_BatchSize=500, P_Epochs=1, C_BatchSize=500, C_Epochs=1)
    
    #Report current Player_One cumulative/aggregate win rate to console
    Game_Meta_pdf = pd.read_csv(Game_Meta_Data_Path)
    Player_One_Win_Rate = Game_Meta_pdf['Winner Model_ID'].value_counts()[Player_One.Model_ID] / len(Game_Meta_pdf)
    print(Player_One.Model_ID + ' Win Rate: ' + str(Player_One_Win_Rate))
    continue
'''

'''
from sklearn.neighbors import KNeighborsRegressor
from matplotlib import pyplot as plt

Project_Base_Path = 'C:/Users/Jake/Documents/Actual/Data Science/Projects/Artificial Intelligence/Specialized/Proof House/Connect Four'
Game_Meta_Data_Path = Project_Base_Path + '/Data/Games Meta Data/Game_Operator_Observations.csv'
Game_Meta_pdf = pd.read_csv(Game_Meta_Data_Path)

Game_IDs = Game_Meta_pdf['Game_ID'].values
Player_One_Wins = Game_Meta_pdf['Player_One Win'].values

k = 2000
KNN_Model = KNeighborsRegressor(n_neighbors=k)
KNN_Model.fit(Game_IDs.reshape(-1,1), Player_One_Wins.reshape(-1,1))

plt.figure()
plt.title('Smoothing Function Bin Width: ' + str(k) + ', Rewards Bandwidth: ' + str(Player_One.Rewards_Bandwidth))
plt.ylabel('Smoothed Player_One Win Rate')
plt.xlabel('Game_ID')
plt.plot(Game_IDs, KNN_Model.predict(Game_IDs.reshape(-1,1)).reshape(-1))
'''