import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Device configuration: checks for GPU support
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 784 # 28x28 - image size
hidden_size = 500
num_classes = 10 # 10 different digit types
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# Apply transform: convert data to flat tensors
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,),(0.3081,))])

# Specify training and test dataset (using MNIST)
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transform,
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transform)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# display sample of data using matplotlib
examples = iter(test_loader)
example_data, example_targets = examples.next()
print (example_data.shape, example_targets.shape)

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(example_data[i][0], cmap='gray')
#plt.show()

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out


model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [100, 1, 28, 28]
        # resized: [100, 784]
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels) #gets predicted outputs and labels

        # Backward and optimize
        optimizer.zero_grad() #empty values in gradient attribute
        loss.backward() #backpropagation
        optimizer.step() #update step - updates parameters

        if (i+1) % 100 == 0: #every 100th step, print current epoch, step, and loss
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad(): #wrap in with statement
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader: #loop through all batches in test samples
        images = images.reshape(-1, 28*28).to(device) #reshape, push to device
        labels = labels.to(device)
        outputs = model(images)
        # torch.max returns (value, index)
        _, predicted = torch.max(outputs.data, 1) #get actual predictions
        n_samples += labels.size(0) #number of samples in current batch
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples #accuracy percentage
    print(f'Accuracy of the network on the 10000 test images: {acc} %')

torch.save(model.state_dict(), "mnist_ffn.pth")
