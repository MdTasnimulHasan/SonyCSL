import numpy as np
from numpy.ma.core import ceil
from scipy.spatial import distance #distance calculation
from sklearn.preprocessing import MinMaxScaler #normalisation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score #scoring
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import animation, colors
import argparse
import pandas as pd
import ast
from sklearn.preprocessing import LabelEncoder
import os

def get_args():
    """
    make parser to get parameters
    """

    parser = argparse.ArgumentParser(
        prog='cVAE',
        description='using cVAE for 768 dimensional data',
        epilog='Example: python cVAE.py --src_filename --output_filename')
    
    parser.add_argument('--src_filename', type = str, default = 'input_data_cvae.tsv', help='source .tsv file')
    parser.add_argument('--output_folder', type = str, default = 'E://Tasnim//SonyCSL//SOM//som_animated', help='output animated plot folder path ')
    

    return parser.parse_args()




args = get_args()

# load data as np array
df_load = pd.read_csv(args.src_filename, sep="\t", header=None)
# column_names =  ['id','embedding','document_type']
headers = df_load.iloc[0]
df  = pd.DataFrame(df_load.values[1:], columns=headers)
# print(df['embedding'])
str_vector = df['embedding']
np_vector_list = []
for i in range (0, len(df['embedding']), 1):
    np_vector_list.append(np.array(ast.literal_eval(str_vector[i])).astype(np.float32))
np_vector = np.array(np_vector_list)
print(np_vector)

# load document type and encode them
label_df = df['document_type']
label_encoder = LabelEncoder()
label = label_encoder.fit_transform(label_df)
print(label)  # Output: [0, 1, 0, 2, 1]


train_x, test_x, train_y, test_y = train_test_split(np_vector, label, test_size=0.2, random_state=42, stratify=label)
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape) # check the shapes



# Helper functions

# Data Normalisation
def minmax_scaler(data):
  scaler = MinMaxScaler()
  scaled = scaler.fit_transform(data)
  return scaled

# Euclidean distance
def e_distance(x,y):
  return distance.euclidean(x,y)

# Manhattan distance
def m_distance(x,y):
  return distance.cityblock(x,y)

# Best Matching Unit search
def winning_neuron(data, t, som, num_rows, num_cols):
  winner = [0,0]
  shortest_distance = np.sqrt(data.shape[1]) # initialise with max distance
  input_data = data[t]
  for row in range(num_rows):
    for col in range(num_cols):
      distance = e_distance(som[row][col], data[t])
      if distance < shortest_distance: 
        shortest_distance = distance
        winner = [row,col]
  return winner

# Learning rate and neighbourhood range calculation
def decay(step, max_steps,max_learning_rate,max_m_distance):
  coefficient = 1.0 - (np.float64(step)/max_steps)
  learning_rate = coefficient*max_learning_rate
  neighbourhood_range = ceil(coefficient * max_m_distance)
  return learning_rate, neighbourhood_range


# hyperparameters
num_rows = 20
num_cols = 20
max_m_distance = 5
max_learning_rate = 0.5
max_steps = int(100*10e3)

# num_nurons = 5*np.sqrt(train_x.shape[0])
# grid_size = ceil(np.sqrt(num_nurons))
# print(grid_size)


#mian function

train_x_norm = minmax_scaler(train_x) # normalisation

# initialising self-organising map
num_dims = train_x_norm.shape[1] # numnber of dimensions in the input data
np.random.seed(40)
som = np.random.random_sample(size=(num_rows, num_cols, num_dims)) # map construction

# start training iterations
for step in range(max_steps):
    if (step+1) % 1000 == 0:
      print("Iteration: ", step+1) # print out the current iteration for every 1k
    learning_rate, neighbourhood_range = decay(step, max_steps,max_learning_rate,max_m_distance)

    t = np.random.randint(0,high=train_x_norm.shape[0]) # random index of traing data
    winner = winning_neuron(train_x_norm, t, som, num_rows, num_cols)
    for row in range(num_rows):
      for col in range(num_cols):
        if m_distance([row,col],winner) <= neighbourhood_range:
          som[row][col] += learning_rate*(train_x_norm[t]-som[row][col]) #update neighbour's weight

    
    
    if (step+1) % 1000 == 0:
        
        # collecting labels
      
        label_data = train_y
        map = np.empty(shape=(num_rows, num_cols), dtype=object)
      
        for row in range(num_rows):
          for col in range(num_cols):
            map[row][col] = [] # empty list to store the label
           
        for t in range(train_x_norm.shape[0]):
            winner = winning_neuron(train_x_norm, t, som, num_rows, num_cols)
            map[winner[0]][winner[1]].append(label_data[t]) # label of winning neuron
          
        
        # construct label map
        label_map = np.zeros(shape=(num_rows, num_cols),dtype=np.int64)
        
        for row in range(num_rows):
            for col in range(num_cols):
                label_list = map[row][col]
                if len(label_list)==0:
                    label = 3  # unassigned
                else:
                    label = max(label_list, key=label_list.count)
                label_map[row][col] = label
                
        unique_labels = np.unique(label_map)
        num_labels = len(unique_labels)
        print(unique_labels, num_labels)
        # # Automatically generate a colormap based on the number of unique labels
        # cmap = plt.cm.get_cmap('tab10', num_labels)
        title = ('Iteration ' + str(step+1))
        cmap = colors.ListedColormap(['tab:green', 'tab:red', 'tab:orange', 'tab:blue'])
        plt.imshow(label_map, cmap=cmap)
        plt.colorbar()
        plt.title(title)
        plt.savefig(os.path.join(args.output_folder,str(step+1)), bbox_inches='tight', dpi =200)
        plt.show()
        

print("SOM training completed")



# collecting labels

label_data = train_y
map = np.empty(shape=(num_rows, num_cols), dtype=object)

for row in range(num_rows):
    for col in range(num_cols):
        map[row][col] = [] # empty list to store the label

for t in range(train_x_norm.shape[0]):
    if (t+1) % 1000 == 0:
        print("sample data: ", t+1)
    winner = winning_neuron(train_x_norm, t, som, num_rows, num_cols)
    map[winner[0]][winner[1]].append(label_data[t]) # label of winning neuron
  
  
# construct label map
label_map = np.zeros(shape=(num_rows, num_cols),dtype=np.int64)
for row in range(num_rows):
    for col in range(num_cols):
        label_list = map[row][col]
        if len(label_list)==0:
            label = -1  # unassigned
        else:
            label = max(label_list, key=label_list.count)
        label_map[row][col] = label

title = ('Iteration ' + str(max_steps))
cmap = colors.ListedColormap(['tab:green', 'tab:red', 'tab:orange', 'tab:blue'])
plt.imshow(label_map, cmap=cmap)
plt.colorbar()
plt.title(title)
plt.show()


#%%

# test data

# using the trained som, search the winning node of corresponding to the test data
# get the label of the winning node

data = minmax_scaler(test_x) # normalisation

winner_labels = []

for t in range(data.shape[0]):
    winner = winning_neuron(data, t, som, num_rows, num_cols)
    row = winner[0]
    col = winner[1]
    predicted = label_map[row][col]
    winner_labels.append(predicted)

print('predicted:')
print(winner_labels)
print('ground truth')
print(list(test_y))
print("Accuracy: ",accuracy_score(test_y, np.array(winner_labels)))




label_data = test_y
map = np.empty(shape=(num_rows, num_cols), dtype=object)

for row in range(num_rows):
  for col in range(num_cols):
    map[row][col] = [] # empty list to store the label
   
for t in range(data.shape[0]):
    winner = winning_neuron(data, t, som, num_rows, num_cols)
    map[winner[0]][winner[1]].append(label_data[t]) # label of winning neuron
  

# construct label map
label_map = np.zeros(shape=(num_rows, num_cols),dtype=np.int64)

for row in range(num_rows):
    for col in range(num_cols):
        label_list = map[row][col]
        if len(label_list)== 0:
            label = 3  # unassigned
        else:
            label = max(label_list, key=label_list.count)
        label_map[row][col] = label
        
unique_labels = np.unique(label_map)
num_labels = len(unique_labels)
print(unique_labels, num_labels)
# # Automatically generate a colormap based on the number of unique labels
# cmap = plt.cm.get_cmap('tab10', num_labels)
title = ('test_map')
cmap = colors.ListedColormap(['tab:green', 'tab:red', 'tab:orange', 'tab:blue'])
plt.imshow(label_map, cmap=cmap)
plt.colorbar()
plt.title(title)
plt.savefig(os.path.join(args.output_folder,'test_map'), bbox_inches='tight', dpi =200)
plt.show()




