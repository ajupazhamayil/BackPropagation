import numpy as np, numpy.random
import csv
import math
from random import shuffle
import logging,sys
from collections import OrderedDict
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
def printList(array):
	for i in range(len(array)):
		print(array[i])
def checkSum2D(array):
	sum=0
	for i in range(len(array)):
		for j in range(len(array[0])):
			sum+=array[i][j]
	return sum
def checkSum(array):
	sum=0
	for i in range(len(array)):
		sum+=array[i]
	return sum
def calculateNetIp(x,node):
	#node=1 for output layer, 0 for hidden layer
	global ip_hl_w,bias,output,net_ip,hl_ol_w
	if node==0:
		for i in range(5):
			sum=0
			for j in range(len(ip_hl_w)):#5 hidden layer nodes
				sum+=ip_hl_w[j][i]*float(x[j])
			sum+=bias[i]
			net_ip[i]=sum
	if node==1:
		index=5#index of output node
		sum=0
		for i in range(len(hl_ol_w)):
			sum+=hl_ol_w[i]*output[i]
		sum+=bias[5]
		net_ip[5]=sum
def calculateOp(node):
	global net_ip,output
	if node==0:
		for i in range(5):
			output[i]=(1/(1+math.exp(-1*net_ip[i])))
	if node==1:
		output[5]=(1/(1+math.exp(-1*net_ip[5])))
def calculateError(node_type,index):
	#node_type=1 for output layer, 0 for hidden layer
	global output,dataset,error,hl_ol_w
	if node_type==1:
		error[5]=output[5]*(1-output[5])*(dataset[index][-1]-output[5])
	if node_type==0:
		for i in range(5):
			error[i]=output[i]*(1-output[i])*error[5]*hl_ol_w[i]
def updateWeights(x):
	global error,output,ip_hl_w,hl_ol_w,bias
	#update hl to ol weights
	change=0
	for i in range(len(hl_ol_w)):#for ol, index is 5
		change=learning_rate*error[5]*output[i]#changed
		hl_ol_w[i]=hl_ol_w[i]+change
	#update ip to hl weights
	for i in range(len(ip_hl_w)):
		for j in range(5):
			change=learning_rate*error[j]*float(x[i])#changed
			ip_hl_w[i][j]=ip_hl_w[i][j]+change
	#update the bias
	for i in range(6):
		bias[i]=bias[i]+learning_rate*error[i]
def confusion_matrix(classes,actual_class,predicted_class):
	#returns TP-1 FN-2 FP-3 TN-4
	global tp,fn,fp,tn
	logging.debug("classes : "+str(classes))
	logging.debug("Actual class : "+str(actual_class))
	logging.debug("Predicted class: "+str(predicted_class))
	if actual_class==predicted_class:
		if actual_class==classes[0]:
			tp+=1
		else:
			tn+=1
	else:
		if actual_class==classes[0]:
			fn+=1
		else:
			fp+=1
choice=int(input("\n1. IRIS\n2. SPECT\n3. SPECTF\n  Select Dataset : "))
if(choice==1):
	file_name="IRIS.csv"
elif(choice==2):
	file_name="SPECT.csv"
elif(choice==3):
	file_name="SPECTF.csv"
else:
	sys.exit()
no_attr=0
with open(file_name, 'r') as f:
	reader = csv.reader(f)
	dataset = list(reader)
if dataset[0][0].strip().lower()=="class":
	for i in range(len(dataset)):
		dataset[i].append(dataset[i].pop(0))
no_attr=len(dataset[0])-1
logging.info("No. of attributes = %s" %(no_attr))
dataset=dataset[1:]

unique_classes=[]
for i in range(len(dataset)):
	unique_classes.append(dataset[i][-1])
unique_classes=list(OrderedDict.fromkeys(unique_classes))
for i in range(len(dataset)):
	if dataset[i][-1]==unique_classes[0]:
		dataset[i][-1]=1
	else:
		dataset[i][-1]=0
#Change class name to 0 or 1
unique_classes[0]=1
unique_classes[1]=0
logging.debug(dataset)
shuffle(dataset)
logging.debug("After shuffle")
logging.debug(dataset)
logging.info("Total length = "+str(len(dataset)))
logging.info("unique_classes "+str(unique_classes))

temp_w=np.random.dirichlet(np.ones(no_attr*5),size=1)[0].tolist()#list of weights for ip to hidden layer
#input to hidden layer weights
ip_hl_w=[temp_w[i*5:(i+1)*5] for i in range(no_attr)]
#hidden layer to op layer weights
hl_ol_w=np.random.dirichlet(np.ones(5),size=1)[0].tolist()
logging.info("Initial weights")
logging.info("hidden layer ")
logging.info(ip_hl_w)
logging.debug("Sum of ip_hl_w matrix :"+str(checkSum2D(ip_hl_w)))
logging.info("hl to output layer : "+str(hl_ol_w))
logging.debug("Sum of hl_ol wights : "+str(checkSum(hl_ol_w)))
#Assign initial bias values to hidden layer nodes and output node
bias=np.random.dirichlet(np.ones(6),size=1)[0].tolist()
logging.info("bias : "+str(bias))
logging.debug("Sum of bias : "+str(checkSum(bias)))
epoch=1000
testing_index=0
learning_rate=0.9
check=[]
block_size=len(dataset)//10
logging.info("Block Size = "+str(block_size))
tp,fn,fp,tn=(0,)*4
for testing_index in range(10):
	training=[]
	net_ip=[0 for i in range(6)]
	output=[0 for i in range(6)]
	error=[0 for i in range(6)]
	if(testing_index!=9):
		testing=[i for i in range(block_size*testing_index,(testing_index+1)*block_size)]
	else:
		testing=[i for i in range(block_size*testing_index,len(dataset))]
	training=[i for i in range(0,testing_index*block_size)]
	if testing_index!=9:
		training+=[i for i in range((testing_index+1)*block_size,len(dataset))]
	for learning_iter in range(epoch):
		global_error=[]
		for train_i in training:
			global_error=[]
			x=dataset[train_i][:no_attr]
			#for each hl node and ol node, calculate net_ip
			calculateNetIp(x,0)
			calculateOp(0)
			#calculate output node net_ip and op
			calculateNetIp(x,1)
			calculateOp(1)
			calculateError(1,train_i)
			calculateError(0,train_i)
			global_error.append(abs(output[5]-dataset[train_i][-1]))
			updateWeights(x)
			logging.debug("net ip : "+str(net_ip))
			logging.debug("output : "+str(output))
		global_error_val=sum(global_error)/float(len(global_error))
		#print(global_error_val)
		#experiment
		if global_error_val<0.001:
			break
	logging.info("Error : ",error)
	for testing_index in testing:
		x=dataset[testing_index][:no_attr]
		#for each hl node and ol node, calculate net_ip
		calculateNetIp(x,0)
		calculateOp(0)
		#calculate output node net_ip and op
		calculateNetIp(x,1)
		calculateOp(1)
		actual_class=dataset[testing_index][-1]
		predicted_class=int(round(output[5]))
		confusion_matrix(unique_classes,actual_class,predicted_class)
		check.append(testing_index)
		#print("output[5]: "+str(output[5])+"   "+str(predicted_class))
logging.info(check)
print("TP = ",tp)
print("FN = ",fn)
print("FP = ",fp)
print("TN = ",tn)
if tp+fp!=0:
	p_precision=tp/(tp+fp)
	print("Positive Precision = ",p_precision)
if tn+fn!=0:
	n_precision=tn/(tn+fn)
	print("Negative Precision = ",n_precision)
if tp+fn!=0:
	p_recall=tp/(tp+fn)
	print("Positive Recall = ",p_recall)
if tn+fp!=0:
	n_recall=tn/(tn+fp)
	print("Negative Recall = ",n_recall)
accuracy=(tp+tn)/(tp+tn+fp+fn)
print("Accuracy = ",accuracy)
