import os
import torch
import torch.nn as nn # from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 1               
BATCH_SIZE = 50
LR = 0.001              
DOWNLOAD_MNIST = False

#MNIST(Modified-NIST): 60,000 trainning images 10,000 testing images. 28*28 pixels with grayscale. Another dataset Extended-MNIST (EMNIST) is larger.

# Mnist digits dataset
dataset_path='../mnist/'
if not(os.path.exists(dataset_path)) or not os.listdir(len([f for f in os.listdir(dataset_path) if not f.startswith('.')])>0):  #os.listdir(path) return a list with all files in the path
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True

train_data = datasets.MNIST(
    root='mnist',
    train=True,                                     
    transform=ToTensor(),                           # https://pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html
                                                    # tranform to tensor and scaled from [0,255] to [0.0,1.0]
    download=DOWNLOAD_MNIST,
)

# plot one example
print(train_data.train_data.size())                 # (60000, 28, 28)
print(train_data.train_labels.size())               # (60000)
plt.imshow(train_data.train_data[0].numpy(), cmap='gray') # grayscale
#plt.imshow(train_data.train_data[0].numpy()) # arbitrary displaying color
plt.title('%i' % train_data.train_labels[0])
plt.show()

# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE,shuffle=True)
'''for multiprocessing:
train_loader = Data.DataLoader(dataset=train_data, 
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                num_workers=args.nThreads)
'''

# pick 2000 samples to speed up testing
test_data = datasets.MNIST(root='mnist', train=False)
#tensor.size() to check shape of a tensor
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels[:2000]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters, each filter is of the shape (kernel_size,kernel_size,int_channels)
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # to make the output the same size as input (H+2P)-kernelsize+1=H ===> P= (5-1)/2=2
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)                   # (batch_size,1,28,28) to (batch_size,16,14,14)   
        x = self.conv2(x)                   # (batch_size,16,14,14) to (batch_size,32,7,7)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)  
        output = self.out(x)                
        return output, x    # return x for visualization


cnn = CNN()
print(cnn)  # net architecture

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

# visualization
from matplotlib import cm
try: from sklearn.manifold import TSNE; HAS_SK = True; 
except: HAS_SK = False; print('Please install sklearn for layer visualization')
def plot_with_labels(lowDWeights, labels):
    plt.cla()     #Clear the current axes.
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9)); # rainbow is one of the colormaps
        plt.text(x, y, s, backgroundcolor=c, fontsize=9) 
    plt.xlim(X.min(), X.max())      #set the left and right x limits of the current axes, or by xlim(right=3). Get by left, right = xlim()  
    plt.ylim(Y.min(), Y.max())
    plt.title('Visualize last layer')
    plt.show()
    plt.pause(0.01)

plt.ion()
# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader

        output = cnn(b_x)[0]            # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        if step % 50 == 0:
            test_output, last_layer = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy() # indices of max
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
            if HAS_SK:
                # Visualization of trained flatten layer (T-SNE) #T-distributed Stochastic Neighbor Embedding t分布随机邻居嵌入
                tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
                plot_only = 500
                low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
                labels = test_y.numpy()[:plot_only]
                plot_with_labels(low_dim_embs, labels)
plt.ioff()

# print 10 predictions from test data
test_output, _ = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy()
print('prediction number:'.ljust(19),pred_y)
print('real number:'.ljust(19),test_y[:10].numpy())
