import numpy as np
from download_mnist import load
import time

x_train, y_train, x_test, y_test = load()
# x_train : 60,000x784 numpy array that each row contains flattened version
# of training images.
# y_train : 1x60,000 numpy array that each component is true label of the
# corresponding training images.
# x_test : 10,000x784 numpy array that each row contains flattened version of
# test images.
# y_test : 1x10,000 numpy array that each component is true label of the
# corresponding test images.

#Prepare for onehot labels:
Actual_Output = np.zeros((60000, 10))
for q in range(60000):
    Actual_Output[q][y_train[q]] = 1

class FC_NN_MNIST:
    def __init__(self, Dim_input, Dim_hid0, Dim_hid1, Dim_out, learning_rate = 0.01,  Mini_batch_size = 128):
        #Variable Declaration
        self.Dim_input = Dim_input
        self.Dim_hid0 = Dim_hid0
        self.Dim_hid1 = Dim_hid1
        self.Dim_out = Dim_out
        self.learning_rate = learning_rate
        self.Mini_batch_size = Mini_batch_size

        #Initaialize weights and bias with random numbers:
        np.random.seed(1)
        # Creating Synapses for  each layers to store weights
        self.weight0 = learning_rate*np.random.randn(Dim_input,Dim_hid0)
        self.bias0 = np.zeros((1, Dim_hid0))
        self.weight1 = learning_rate*np.random.randn(Dim_hid0,Dim_hid1)
        self.bias1 = np.zeros((1, Dim_hid1))
        self.weight2 = learning_rate*np.random.randn(Dim_hid1,Dim_out)
        self.bias2 = np.zeros((1, Dim_out))


    #Relu Activation function:
        #When derv = False: It is doing forward pass calculation
        #When derv = True: It is doing backward pass calculation

        #Forward Pass:
            #Relu = max(0,x)
        #Backward Pass:
            #Relu = {0 if x<0 or 1 otherwise}
    def Relu(self, y ,derv=False):
            x = y.copy()
            if(derv == False):
                return np.maximum(0,x)

            i, j = x.shape
            for k in range(i):
                for l in range(j):
                    if(x[k][l] <= 0):
                        x[k][l] = 0
                    else:
                        x[k][l] = 1
            return x


    # Softmax Activation function for Output layer
    def SoftMax(self, y, derv=False):
        x = y.copy()
        row, col = x.shape
        if(derv == False):
            for r in range(row):
                x[r] -=np.max(x[r])
                x[r] = np.exp(x[r])/np.sum(np.exp(x[r]))
            return x

        for r in range(row):
            x[r] -= np.max(x[r])
            st1 = (np.exp(x[r])/np.sum(np.exp(x[r])))
            st2 = (1-(np.exp(x[r])/np.sum(np.exp(x[r]))))
            x[r] = st1*st2
        return x


    def CrossEntropyLoss(self, pred, real):
        temp1 = np.log2(pred)
        return (-1*np.sum((temp1*real)))/real.shape[0]

    #compute dL/dz3::
    def Compute_Gradiant(sef, pred, real):

        row, col = real.shape
        temp = pred
        temp = temp - real
        return temp/real.shape[0]

    def Forward_Propogation(self, Input):
        z1 = np.dot(Input, self.weight0)  + self.bias0
        Hidden_Layer1 = self.Relu(z1)
        z2 = np.dot(Hidden_Layer1, self.weight1) + self.bias1
        Hidden_Layer2 = self.Relu(z2)
        z3 = np.dot(Hidden_Layer2, self.weight2) + self.bias2
        Output =  self.SoftMax(z3)
        return Hidden_Layer1, Hidden_Layer2, Output

    def Train(self,epoch_no,Training_data, Training_output):
        Total_batches = int(np.ceil(Training_data.shape[0]/self.Mini_batch_size))
        for batch_idx in range(Total_batches):
            if(batch_idx == Total_batches-1):
                Input = Training_data[(Total_batches-1)*self.Mini_batch_size : Training_data.shape[0]]
                True_Output = Training_output[(Total_batches-1)*self.Mini_batch_size : Training_data.shape[0]]
            else:
                Input = Training_data[batch_idx*self.Mini_batch_size : (batch_idx+1)*self.Mini_batch_size]
                True_Output = Training_output[batch_idx*self.Mini_batch_size : (batch_idx+1)*self.Mini_batch_size]

            #Forward Pass:
            Hidden_Layer1, Hidden_Layer2, Output = self.Forward_Propogation(Input)

            # Compute Loss:
            #Cross Entropy Loss:
            Loss = self.CrossEntropyLoss(Output, True_Output)

            # Input --w0--> [H1:  (z1 = Input*weight0+bias0 | Hidden_Layer1 = Relu(z1))] --w1-->
            #   [H2:  (z2 = Hidden_Layer1*weight1+bias1 | Hidden_Layer2 = Relu(z2))] --w2-->
            #   [Out: (z3 = Hidden_Layer2*weight2 +bias2 | Output  = SoftMax(z3))]
            #Compute the gradiant on output probability:
            #dL/dz3 = Derivative of Cross CrossEntropyLoss respect to softmax * Derivative of softmax respect to z3
            dz3 = self.Compute_Gradiant(Output, True_Output) # [dL/d(Output)]*[d(Output)/d(z3)] = dL/d(z3)
            #H2:
            dHidden_layer2 = np.dot(dz3, self.weight2.T) # dL/d(Hidden_Layer2) = [dL/d(z3)]*[d(z3)/d(Hidden_Layer2)]
            self.weight2 -= self.learning_rate*np.dot(Hidden_Layer2.T, dz3) #dL/dw2 = [dL/d(Output)]*[d(Output)/d(z3)]*[d(z3)/d(dw2)]
            self.bias2 -= self.learning_rate *np.sum(dz3,axis=0, keepdims=True) #bias2
            dz2 = dHidden_layer2*self.Relu(Hidden_Layer2,derv=True) #dL/d(z2) = [dL/d(Hidden_Layer2)]*[d(Hidden_Layer2)/d(z2)]

            #H1:
            dHidden_Layer1 = np.dot(dz2, self.weight1.T) # dL/d(Hidden_Layer1) = [dL/dz2]*[dz2/d(Hidden_layer1)]
            self.weight1 -= self.learning_rate*np.dot(Hidden_Layer1.T, dz2) # dL/dw1 = [dL/d(Hidden_Layer2)]*[d(Hidden_Layer2)/d(z2)]*[d(z2)/dw2]
            self.bias1 -= self.learning_rate*np.sum(dz2, axis=0)#bias1
            dz1 = dHidden_Layer1*self.Relu(Hidden_Layer1, derv=True) #dL/dz1 = [dL/d(Hidden_Layer1)]*[d(Hidden_Layer1)/dz1]

            #Input:
            self.weight0 -= self.learning_rate*np.dot(Input.T, dz1)#dL/dw0 = [dL/dz1]*[dz1/dw0]
            self.bias0 -= self.learning_rate*np.sum(dz1, axis=0)# bias0

            if(batch_idx % 10) == 0:
                print('Train Epoch: {} [{}/{} ({:.3f}%)]\tError: {:.6f}'.format(epoch_no, batch_idx*len(Input), len(Training_data), 100.*(batch_idx+1)*len(Input)/len(Training_data), Loss ))

    def Test(self, Testing_data, Testing_output):
        #Forward Pass:
        H1, H2, Test_Output = self.Forward_Propogation(Testing_data)
        correct = 0
        for w in range(Testing_data.shape[0]):
            if(np.argmax(Test_Output[w]) == Testing_output[w]):
                correct =correct +1
        print('Test Accuracy:{}/{} ({:.3f}%)\n'.format( correct, len(Testing_data), 100.*correct/len(Testing_data)))


if __name__ == "__main__":
    epochs = 10
    #The model size = 784-200-50-10
    MNIST = FC_NN_MNIST(784,200,50,10)
    #Start Training:
    time0 = time.time()
    start_time = time.time()
    Total_Training_Time = 0
    for epoch in range(1, epochs+1):
        MNIST.Train(epoch, x_train,Actual_Output)
        Total_Training_Time += (time.time() - start_time)
        print("TRAIN DATA TESTING ACCURACY:")
        MNIST.Test(x_train, y_train)
        print("TEST DATA TESTING ACCURACY: ")
        MNIST.Test(x_test, y_test)
        print("Training Time: ", Total_Training_Time, " secs")
        print("Execution Time: ", time.time()-time0, " secs")
        start_time = time.time()
