import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import functional as F
import tqdm

trainData = pd.read_csv('/Users/camillegagne/desktop/cropharvest-crop-detection/train.csv')
testData = pd.read_csv('/Users/camillegagne/desktop/cropharvest-crop-detection/test_nolabels.csv')

testData = torch.FloatTensor(testData.to_numpy())

#Separation des labels et des features et creation d'un set de test et validation
df = pd.DataFrame(trainData)
data = df.iloc[:, 0:-1]
labels = df.iloc[:, -1]
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.20, random_state=69)

val = torch.FloatTensor(y_val.to_numpy())

#Normalizing
train_data = StandardScaler().fit_transform(X_train)
valid_data = StandardScaler().fit_transform(X_val)

#Transformations en tensor
train_data = torch.FloatTensor(train_data)
y_train = torch.FloatTensor(y_train)
valid_data = torch.FloatTensor(valid_data)


class Net(nn.Module):
	
	def __init__(self,input_shape:int = 217, 
					lr: float = 0.0001, 
					epoch: int=2,
					batch_size: int=500):
		super(Net,self).__init__()
		self.lr = lr
		self.epoch = epoch
		self.batch_size = batch_size
		self.input_shape = input_shape
		self.epsilon = 1e-9
		self.train_data = train_data
		self.network = self.forward(self.train_data)
	
	def forward(self,x):
		return nn.Sequential(
			nn.Linear(self.input_shape,256),
			nn.ReLU(),
			nn.Linear(256,128),
			nn.ReLU(),
			nn.Linear(128,1),
			nn.Sigmoid()
			)
		
	def train_steps(self, X_batch, y_batch):
		model = Net(self.input_shape)
		optimizer = torch.optim.SGD(model.parameters(),lr=self.lr)
		optimizer.zero_grad()
		loss, accuracy = self.compute_loss_and_accuracy(X_batch, y_batch)
		loss.backward()
		optimizer.step()
		return accuracy

	def compute_loss_and_accuracy(self, X, y):
		pred = self.network(X)
		clamped = torch.clamp(pred, self.epsilon, 1 - self.epsilon) # clamp to [epsilon, 1-epsilon]
		BCE = nn.BCELoss()
		loss = BCE(torch.flatten(clamped), y)
		pred = torch.flatten(pred)
		pred = torch.as_tensor((pred - 0.5) > 0, dtype=torch.int32)	
		f1 = f1_score(pred, y)
		#accuracy = ((pred == y).sum().item() / y.size(dim=0))*100

		return (loss, f1)	

	def train_loop(self):
		n_batches = int(np.ceil(train_data.shape[0] / self.batch_size))

		acc = []
		#tqdm c'est la bar de progres qui apparait dans le terminal,. tu peux ignorer
		for e in tqdm.tqdm(range(n.epoch)):
			for batch in range(n_batches):
				minibatchX = train_data[self.batch_size * batch:self.batch_size * (batch + 1), :]
				minibatchY = y_train[self.batch_size * batch:self.batch_size * (batch + 1)]
				accuracy = self.train_steps(minibatchX, minibatchY)
				acc.append(accuracy)

		return acc

n = Net(217, 0.0001, 1)
trainAccuracy = n.train_loop()

''' Si tu veux voir le accuracy pour chaque step
for i in trainAccuracy:
	print(int(i))	
'''
trainPred = n.compute_loss_and_accuracy(train_data, y_train)
validPred = n.compute_loss_and_accuracy(valid_data, val)
print("Training F1 Score: ", trainPred[1])
print("Validation F1 Score:", validPred[1])


####################### TEST PREDICTION #######################
#Make into dataframe and export to csv

testPred = n.network(testData)
testPred = torch.as_tensor((torch.flatten(testPred) - 0.5) > 0, dtype=torch.int32)	
#print(testPred[:25])

sub = pd.DataFrame(data=testPred, columns=["LABELS"])
sub.index.name = "S.No"
#UPDATE PATH AND NAME 
sub.to_csv('/Users/camillegagne/desktop/sub.csv')








