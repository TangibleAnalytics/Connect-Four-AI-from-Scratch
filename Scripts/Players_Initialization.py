# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 19:07:14 2019

@author: Jake
"""

import pickle
import Model_Architectures as MA
import Connect_Four as CF

Project_Base_Path = 'C:/Users/Jake/Documents/Actual/Data Science/Projects/Artificial Intelligence/Specialized/Proof House/Connect Four'


#Player One
Player_One_Train_Data_Path = Project_Base_Path + '/Data/Player One/Training Data/Player_Training_Observations.csv'
Rewards_Bandwidth = 5
Perception_Model = MA.Perception_Model_Definition(Board_Shape=(6,7), Action_Space=7)
Policy_Model = MA.Policy_Model_Definition(Board_Shape=(6,7), Action_Space=7)
Model_ID = 'Chad_0'
Player_One = CF.AI_Player(Player_One_Train_Data_Path, Rewards_Bandwidth, Perception_Model, Policy_Model, Model_ID, Action_Space=7)
Player_Path = Project_Base_Path + '/Data/Player One/Versions'
pickle.dump(Player_One, open(Player_Path + '/' + Model_ID + '.p', 'wb'))


#Player Two
Player_Two_Train_Data_Path = Project_Base_Path + '/Data/Player Two/Training Data/Player_Training_Observations.csv'
Rewards_Bandwidth = 1
Perception_Model = MA.Perception_Model_Definition(Board_Shape=(6,7), Action_Space=7)
Policy_Model = MA.Policy_Model_Definition(Board_Shape=(6,7), Action_Space=7)
Model_ID = 'JT_0'
Player_Two = CF.AI_Player(Player_Two_Train_Data_Path, Rewards_Bandwidth, Perception_Model, Policy_Model, Model_ID, Action_Space=7)
Player_Path = Project_Base_Path + '/Data/Player Two/Versions'
pickle.dump(Player_Two, open(Player_Path + '/' + Model_ID + '.p', 'wb'))



