# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 22:10:46 2019

@author: Jake
"""

import time
import numpy as np
import csv
import pandas as pd
import json
from keras.utils import to_categorical
from sklearn.metrics import r2_score
from Classification_Metrics import Entropy_Explained
from sklearn.preprocessing import StandardScaler

class Game_Board():
    
    def __init__(self, Rows=6, Columns=7):
        self.Rows = Rows
        self.Columns = Columns
        self.Board = np.zeros((Rows, Columns))
        return
    
    def Execute_Chip_Transaction(self, Column_Selection=None, Chip_Value=None):
        
        #Boolean of whether column was selected or not
        if Column_Selection == None:
            return False
        if Chip_Value == None:
            return False
        
        #Loop through rows from physical high to low
        Row_Index = 0
        while Row_Index < self.Rows:
            
            Grid_Value = self.Board[Row_Index, Column_Selection]
            if Grid_Value == 0:
                Row_Index += 1
                continue
            
            #Otherwise, if it is not equal to zero
            if Grid_Value != 0:
                
                #Column has already been filled
                if Row_Index == 0:
                    return False
                
                #Modify chip value of spot above where we found the current top chip
                self.Board[Row_Index-1, Column_Selection] = Chip_Value
                return True
            
            Row_Index += 1
            continue
        
        #Place chip at the bottom of the column if no other chips found
        self.Board[self.Rows-1, Column_Selection] = Chip_Value
        return True
    
    
    def Evaluate_Board(self):
        #Check in all directions for four chips connected together
        
        #Return zero if a tie
        if (self.Board != 0).all():
            return 0
        
        #Horizontal
        Horizontal_Winner = self.__Horizontal()
        if Horizontal_Winner != 0:
            return Horizontal_Winner
        
        #Work smarter, not harder
        self.Board = np.transpose(self.Board)
        
        #Vertical
        Vertical_Winner = self.__Horizontal()
        if Vertical_Winner != 0:
            self.Board = np.transpose(self.Board)
            return Vertical_Winner
        
        #Do NOT forget to undo the convenient transpose of the board game
        self.Board = np.transpose(self.Board)
        
        #Forward Diagonals
        Diagonal_Winner = self.__Diagonal()
        if Diagonal_Winner != 0:
            return Diagonal_Winner
        
        #Work smarter, not harder
        self.Board = np.flip(self.Board, axis=1)
        
        #Backward Diagonals
        Diagonal_Winner = self.__Diagonal()
        if Diagonal_Winner != 0:
            self.Board = np.flip(self.Board, axis=1)
            return Diagonal_Winner
        
        #Do NOT forget to undo the convenient flipping of the board game
        self.Board = np.flip(self.Board, axis=1)
        
        
        #No more places to put chips and therefore a tie!
        if (self.Board != 0).all():
            return 0
        
        #Return None if no winner has been found
        return None
    
    
    def __Horizontal(self):
        
        for Row_Index in range(self.Board.shape[0]):
            for Column_Index in range(self.Board.shape[1] - 3):
                
                #Grid_Value is zero if no chip placed, or +1/-1 based upon which player placed it there
                Grid_Value = self.Board[Row_Index, Column_Index]
                
                #Continue to next grid space if no chip has been placed in this one
                if Grid_Value == 0:
                    continue
                
                #Otherwise, check next four Grid_Values to the right for equality
                Winner_Result = True
                for Column_Test_Index in range(Column_Index + 1, Column_Index + 4):
                    Grid_Test_Value = self.Board[Row_Index, Column_Test_Index]
                    if Grid_Test_Value != Grid_Value:
                        Winner_Result = False
                        break
                    continue
                
                #If the Winner_Result value has been unchanged
                if Winner_Result == True:
                    return Grid_Value
                
                continue
            continue
        
        #If Horizontal Win not found
        return False
    
    
    def __Diagonal(self):
        
        for Row_Index in range(self.Board.shape[0] - 3):
            for Column_Index in range(self.Board.shape[1] - 3):
                
                #Grid_Value is zero if no chip placed, or +1/-1 based upon which player placed it there
                Grid_Value = self.Board[Row_Index, Column_Index]
                
                #Continue to next grid space if no chip has been placed in this one
                if Grid_Value == 0:
                    continue
                
                #Otherwise, check next four Grid_Values going diagonally down to right for equality
                Winner_Result = True
                for Diagonal_Offset in range(1,4):
                    Grid_Test_Value = self.Board[Row_Index + Diagonal_Offset, Column_Index + Diagonal_Offset]
                    if Grid_Test_Value != Grid_Value:
                        Winner_Result = False
                        break
                    continue
                
                if Winner_Result == True:
                    return Grid_Value

                continue
            continue
        
        #If Diagonal Win not found
        return False
    
    


class Game_Operator():
    
    def __init__(self, Player_One, Player_Two, Game_Meta_Data_Path, Game_ID=None):
        
        #Meta data path
        self.Game_Meta_Data_Path = Game_Meta_Data_Path
        
        #Look up highest Game_ID and increment further for this one
        if Game_ID == None:
            Game_Meta_pdf = pd.read_csv(self.Game_Meta_Data_Path)
            self.Game_ID = max(Game_Meta_pdf['Game_ID']) + 1
        
        #Store players as class attributes
        #Initialize player chip values
        #Initialize each player's Move_ID to zero
        self.Player_One = Player_One
        self.Player_Two = Player_Two
        self.Player_One.Chip_Value = 1
        self.Player_Two.Chip_Value = -1
        self.Player_One.Move_ID = 0
        self.Player_Two.Move_ID = 0
        
        #Initialize a game board
        self.Game_Board = Game_Board(Rows=6,Columns=7)
        
        #Initialize counter for total number of moves taken
        self.Total_Moves = 0
        
        #Initialize winner/loser Model_ID values to zero
        self.Winner_Model_ID = 0
        self.Loser_Model_ID = 0
        return
    
    def Execute_Round(self):
        
        #Record game start time
        self.Start_Time = pd.to_datetime((10**9) * time.time()).strftime(format='%m-%d-%Y %H:%M:%S')
        
        #Relay Game_ID to each player
        self.Player_One.Game_ID = self.Game_ID
        self.Player_Two.Game_ID = self.Game_ID
        
        #Simulate Coin Flip
        Coin_Value = np.random.randint(low=0, high=2)
        
        #Starting Move
        if Coin_Value == 0:
            #Player_Two goes first
            self.__Facilitate_Move(Player_Object=self.Player_Two)
            self.Total_Moves += 1
        
        #Can safely assume that neither has won after first move alone
        self.Player_One.State_of_Reward = 0
        self.Player_Two.State_of_Reward = 0
        
        #Play until one player wins or if the entire game board has been filled with chips
        while True:
            
            #Check for a tie game
            #More efficient if code is kept here
            if (self.Game_Board.Board != 0).all():
                Winning_Chip_Value = 0
                break
            
            #Player_One turn
            Winning_Chip_Value = self.__Facilitate_Move(Player_Object=self.Player_One)
            self.Total_Moves += 1
            if Winning_Chip_Value != None:
                break
            
            #PLayer_Two turn
            Winning_Chip_Value = self.__Facilitate_Move(Player_Object=self.Player_Two)
            self.Total_Moves += 1
            if Winning_Chip_Value != None:
                break
            
            #State of Reward is zero for both, because neither has been pronounced the winner
            self.Player_One.State_of_Reward = 0
            self.Player_One.State_of_Reward = 0
            continue
        
        #Allocate bulk reward values
        if Winning_Chip_Value == 0:
            self.Player_One.State_of_Reward = 0
            self.Player_Two.State_of_Reward = 0
            self.Winner_Model_ID = ''
            self.Loser_Model_ID = ''
        elif Winning_Chip_Value == self.Player_One.Chip_Value:
            self.Player_One.State_of_Reward = 1
            self.Player_Two.State_of_Reward = -1
            self.Winner_Model_ID = self.Player_One.Model_ID
            self.Loser_Model_ID = self.Player_Two.Model_ID
        elif Winning_Chip_Value == self.Player_Two.Chip_Value:
            self.Player_One.State_of_Reward = -1
            self.Player_Two.State_of_Reward = 1
            self.Winner_Model_ID = self.Player_Two.Model_ID
            self.Loser_Model_ID = self.Player_One.Model_ID
        
        #Cue each player to record a final move by proxy of choosing an action
        self.Player_One.Choose_Action(Board_State=self.Game_Board.Board)
        self.Player_Two.Choose_Action(Board_State=self.Game_Board.Board)
        
        #Record game end time
        self.End_Time = pd.to_datetime((10**9) * time.time()).strftime(format='%m-%d-%Y %H:%M:%S')
        self.__Record_Observation()
        return
    
    def __Facilitate_Move(self, Player_Object):
        
        #Select column based on Player_One's policy model
        Column_Selection_Index = Player_Object.Choose_Action(Board_State=self.Game_Board.Board)
        
        #Execute Chip Transaction
        Chip_Transaction_Reponse = self.Game_Board.Execute_Chip_Transaction(Column_Selection=Column_Selection_Index, Chip_Value=Player_Object.Chip_Value)
        while Chip_Transaction_Reponse == False:
            Column_Selection_Index = np.random.randint(low=0, high=self.Game_Board.Board.shape[1])
            Chip_Transaction_Reponse = self.Game_Board.Execute_Chip_Transaction(Column_Selection=Column_Selection_Index, Chip_Value=Player_Object.Chip_Value)
            continue
        
        #Evaluate whether there is a winner or not
        Winning_Chip_Value = self.Game_Board.Evaluate_Board()
        return Winning_Chip_Value
    
    def __Record_Observation(self):
        
        #Append each data component to the overall observation
        Observation = []
        Observation.append(self.Game_ID)
        Observation.append(self.Start_Time)
        Observation.append(self.End_Time)
        Observation.append(self.Total_Moves)
        Observation.append(self.Player_One.Model_ID)
        Observation.append(self.Player_Two.Model_ID)
        Observation.append(self.Winner_Model_ID)
        Observation.append(self.Loser_Model_ID)
        
        #Open meta data csv to append the observation... then close to confirm
        self.Meta_Data_File_Handle = open(self.Game_Meta_Data_Path, 'a', newline='')
        CSV_Writer = csv.writer(self.Meta_Data_File_Handle)
        CSV_Writer.writerow(Observation)
        self.Meta_Data_File_Handle.close()
        return
    
    

class AI_Player():
    
    def __init__(self, Train_Data_Path, Rewards_Bandwidth, Perception_Model, Policy_Model, Model_ID, Action_Space=7):
        
        self.Train_Data_Path = Train_Data_Path
        self.Rewards_Bandwidth = Rewards_Bandwidth
        self.Perception_Model = Perception_Model
        self.Policy_Model = Policy_Model
        self.Model_ID = Model_ID
        
        self.Game_ID = 0
        self.Move_ID = 0
        self.Chip_Value = 0
        self.State_of_Reward = 0
        self.Action_Space = Action_Space
        self.Perception_Q2 = 0
        self.Policy_Q2 = 0
        self.Random_Action_Inflation = 0.25
        self.Random_Action_Chance = 1
        return
    
    def Choose_Action(self, Board_State):
        self.Board_State = np.copy(Board_State)
        
        #Avoid unstable results
        Deliberate_Action_Score = np.random.uniform()
        if Deliberate_Action_Score > self.Random_Action_Chance:
            Board_State_Input = self.Board_State.reshape(1, self.Board_State.shape[0], self.Board_State.shape[1])
            self.Chosen_Action = np.argmax(self.Policy_Model.predict(Board_State_Input)[0])
        else:
            self.Chosen_Action = np.random.randint(low=0, high=self.Action_Space)
        
        self.__Record_Observation()
        self.Move_ID += 1
        return self.Chosen_Action
    
    def __Record_Observation(self):
        
        #Append each data component to the overall observation
        Observation = []
        Observation_Timestamp = pd.to_datetime((10**9) * time.time()).strftime(format='%m-%d-%Y %H:%M:%S')
        Observation.append(Observation_Timestamp)
        Observation.append(self.Game_ID)
        Observation.append(self.Move_ID)
        Observation.append(self.Model_ID)
        Observation.append(self.State_of_Reward)
        Observation.append(json.dumps(self.Board_State.tolist()))
        Observation.append(self.Chosen_Action)
        
        #Open trian data csv to append the observation... then close to confirm
        self.Train_Data_File_Handle = open(self.Train_Data_Path, 'a', newline='')
        CSV_Writer = csv.writer(self.Train_Data_File_Handle)
        CSV_Writer.writerow(Observation)
        self.Train_Data_File_Handle.close()
        return
    
    def Learn(self, P_BatchSize, P_Epochs, C_BatchSize, C_Epochs):
        Board_State_X = self.__Perceive(Batch_Size=P_BatchSize, Epochs=P_Epochs, Verbose=1, Validate_Model=True)
        Optimal_Action_Y = self.__Reflect(Board_State_X, Verbose=1)
        self.__Consolidate(Board_State_X, Optimal_Action_Y, Batch_Size=C_BatchSize, Epochs=C_Epochs, Verbose=1, Validate_Model=True)
        return
    
    def __Perceive(self, Batch_Size=1, Epochs=1, Verbose=1, Validate_Model=True):
        Train_pdf = pd.read_csv(self.Train_Data_Path)
        
        #Add 4 columns to Train_pdf... Max_Move_ID, Bulk_Reward, Moves_Until_Finish, Action_Value
        #Remove final move from each game because it is only necessary for measuring reward
        #Remove all observations with moves farther than twice bandwidth from final move
        Game_Agg_pdf = Train_pdf[['Game_ID','Move_ID']].groupby('Game_ID').max()
        Game_Agg_pdf = pd.merge(left=Game_Agg_pdf, right=Train_pdf[['Game_ID','Move_ID','Pre-action Reward State']], on=['Game_ID','Move_ID'])
        Game_Agg_pdf.columns = ['Game_ID','Max_Move_ID','Bulk_Reward']
        Train_pdf = pd.merge(left=Train_pdf, right=Game_Agg_pdf, on='Game_ID')
        Train_pdf['Moves_Until_Finish'] = Train_pdf['Max_Move_ID'] - Train_pdf['Move_ID']
        Train_pdf['Action_Value'] = Train_pdf['Bulk_Reward'] * np.e ** (-1 * Train_pdf['Moves_Until_Finish'] / self.Rewards_Bandwidth)
        Train_pdf = Train_pdf[Train_pdf['Move_ID'] != Train_pdf['Max_Move_ID']]
        Train_pdf = Train_pdf[np.abs(Train_pdf['Action_Value']) > 0.1]
        
        #Sort by some time proxy
        #sort primarily by Game_ID & then by Move_ID
        #Necessary for ensuring that test set is always a most future set of observations
        #In other words, it has NOT been used for training purposes yet
        
        
        #Convert string representation of board states to unifying 3D numpy array
        Board_State_X = []
        for Row_Index in range(len(Train_pdf)):
            Obs_Board_State_X = json.loads(Train_pdf['Pre-action Board State'].iloc[Row_Index])
            Board_State_X.append(Obs_Board_State_X)
            continue
        Board_State_X = np.array(Board_State_X)
        
        #Vectorize the categorical action space
        Action_Taken_X = np.array(Train_pdf['Action Taken'])
        Action_Taken_X = to_categorical(Action_Taken_X)
        
        #Standardize
        Action_Value_Y = np.array(Train_pdf['Action_Value'])
        Y_Scaler = StandardScaler()
        Action_Value_Y = Y_Scaler.fit_transform(Action_Value_Y.reshape(-1,1)).reshape(-1)
        
        
        if Validate_Model == True:
            #Train/Test split the data for validation purposes
            All_Indices = np.linspace(start=0,stop=(len(Train_pdf)-1), num=len(Train_pdf))
            All_Indices = np.array(All_Indices, dtype='int')
            Split_Index = int((0.75) * len(All_Indices))
            Train_Indices = All_Indices[:Split_Index]
            Test_Indices = All_Indices[Split_Index:]
            Board_State_X_Train = Board_State_X[Train_Indices]
            Board_State_X_Test = Board_State_X[Test_Indices]
            Action_Taken_X_Train = Action_Taken_X[Train_Indices]
            Action_Taken_X_Test = Action_Taken_X[Test_Indices]
            Action_Value_Y_Train = Action_Value_Y[Train_Indices]
            Action_Value_Y_Test = Action_Value_Y[Test_Indices]
            
            #Ad Hoc Analysis
            #np.random.shuffle(Action_Taken_X_Train)
            
            #Test model after each epoch
            for Epoch in range(Epochs):
                
                #Train & Test the model
                self.Perception_Model.fit(x=[Board_State_X_Train, Action_Taken_X_Train], y=Action_Value_Y_Train, batch_size=Batch_Size, epochs=1, verbose=Verbose)
                Action_Value_Y_Pred = self.Perception_Model.predict(x=[Board_State_X_Test, Action_Taken_X_Test])
                self.Perception_Q2 = r2_score(Action_Value_Y_Test, Action_Value_Y_Pred)
                
                #Update random action chance based on certainty of policy model
                self.Random_Action_Chance = (self.Random_Action_Inflation) * (1 - self.Perception_Q2) * (1 - self.Policy_Q2)

                #Report validation metrics to terminal
                if Verbose != 0:
                    print(str(self.Model_ID)+' Perception Model Validation Q2: '+str(self.Perception_Q2))
                    print('Finished Epoch: ' + str(Epoch))
                
                continue
            
        elif Validate_Model == False:
            #Just train the model
            self.Perception_Model.fit(x=[Board_State_X, Action_Taken_X], y=Action_Value_Y, batch_size=Batch_Size, epochs=Epochs, verbose=Verbose)
        
        #Return numpy representation of Board_State training data for reflection step
        return np.copy(Board_State_X)
    
    
    def __Reflect(self, Board_State_X, Verbose=1):
        #Copy for plurality purposes
        Board_States_X = np.copy(Board_State_X)
        del (Board_State_X)
        
        #Test each action space column at a time for all observations
        Action_Values_Matrix = np.zeros(shape=(Board_States_X.shape[0], 0))
        for Action_Space_Index in range(self.Action_Space):
            Guess_Actions_Y = np.zeros(shape=(Board_States_X.shape[0], self.Action_Space))
            Guess_Actions_Y[:, Action_Space_Index] = 1
            Guess_Action_Values = self.Perception_Model.predict(x=[Board_States_X, Guess_Actions_Y])
            Action_Values_Matrix = np.concatenate((Action_Values_Matrix, Guess_Action_Values), axis=1)
            
            if Verbose != 0:
                Percent_Complete = round(100 * (Action_Space_Index+1)/self.Action_Space, 2)
                print(str(self.Model_ID)+' Reflection Process '+ str(Percent_Complete) +'% Complete')
            continue
        Optimal_Actions_Y = np.argmax(Action_Values_Matrix, axis=1)
        
        #Vectorize and append columns of zeros for the rare circumstance
        Optimal_Actions_Y = to_categorical(Optimal_Actions_Y)
        while Optimal_Actions_Y.shape[1] < self.Action_Space:
            Optimal_Actions_Y = np.concatenate((Optimal_Actions_Y, np.zeros(shape=(Optimal_Actions_Y.shape[0], 1))), axis=1)
            continue
        return np.array(Optimal_Actions_Y, dtype='float64')
    
    
    def __Consolidate(self, Board_State_X, Optimal_Action_Y, Batch_Size=1, Epochs=1, Verbose=1, Validate_Model=True):
        
        if Validate_Model == True:
            #Train/Test split the data for validation purposes
            All_Indices = np.linspace(start=0,stop=(len(Board_State_X)-1), num=len(Board_State_X))
            All_Indices = np.array(All_Indices, dtype='int')
            Split_Index = int((0.75) * len(All_Indices))
            Train_Indices = All_Indices[:Split_Index]
            Test_Indices = All_Indices[Split_Index:]
            Board_State_X_Train = Board_State_X[Train_Indices]
            Board_State_X_Test = Board_State_X[Test_Indices]
            Optimal_Action_Y_Train = Optimal_Action_Y[Train_Indices]
            Optimal_Action_Y_Test = Optimal_Action_Y[Test_Indices]
            
            #Test model after each epoch
            for Epoch in range(Epochs):
                
                #Train & Test the model
                self.Policy_Model.fit(x=Board_State_X_Train, y=Optimal_Action_Y_Train, batch_size=Batch_Size, epochs=1, verbose=Verbose)
                Optimal_Action_Y_Pred = self.Policy_Model.predict(x=Board_State_X_Test)
                self.Policy_Q2 = Entropy_Explained(Optimal_Action_Y_Test, Optimal_Action_Y_Pred)
                
                #Update random action chance based on certainty of policy model
                self.Random_Action_Chance = (self.Random_Action_Inflation) * (1 - self.Perception_Q2) * (1 - self.Policy_Q2)
                
                #Report validation metrics to terminal
                if Verbose != 0:
                    print(str(self.Model_ID)+' Policy Model Validation Q2: '+str(self.Policy_Q2))
                    print('Finished Epoch: ' + str(Epoch))
                
                continue
        else:
            #Just train the model
            self.Policy_Model.fit(x=Board_State_X, y=Optimal_Action_Y, batch_size=Batch_Size, epochs=Epochs, verbose=Verbose)
        return

