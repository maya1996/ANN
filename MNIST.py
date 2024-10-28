import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as f
from torch.optim.lr_scheduler import StepLR

#data handling, the data is downloaded and test data is spilted to perform training and validating
def data_handling():
    transform = transforms.Compose([transforms.ToTensor()])
    training_data = MNIST(root='./data',train=True,download=True,transform=transform)
    test_data = MNIST(root='./data',train=False,download=True,transform=transform)
    print(training_data)
    print(test_data)

    training_size = len(training_data) - 10000
    validation_size = len(training_data) - training_size
    train_d, validation_d = random_split(training_data, [training_size, validation_size])
    return train_d, validation_d, test_data


def data_load(training_data, valid_data, test_data):
    train = DataLoader(training_data, batch_size=32, shuffle=True)
    validation = DataLoader(valid_data, batch_size=32, shuffle=True)
    testing = DataLoader(test_data, batch_size=32, shuffle=True)
    return train, validation, testing

class ANNMODEL(nn.Module):
    def __init__(self):
        super(ANNMODEL, self).__init__()

        self.hidden_1 = nn.Linear(784, 512)
        self.hidden_2 = nn.Linear(512, 256)
        self.hidden_3 = nn.Linear(256,64)
        self.output = nn.Linear(64, 10)
        self.rl = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.rl(self.hidden_1(x))
        x = self.rl(self.hidden_2(x))
        x = self.rl(self.hidden_3(x))
        x = self.output(x)
        final_output = f.log_softmax(x, dim=1)
        return final_output

def training(num_epochs, model, data_batch, opt, valid_batch):
    train_losses = []
    val_losses = []
    accuracies = []
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, labels in data_batch:
            output = model(images)
            loss = loss_fn(output, labels)
            train_loss += loss.item()

            opt.zero_grad()
            loss.backward()
            opt.step()

        avg_trl = train_loss / len(data_batch)
        train_losses.append(avg_trl)

        val_loss, accu = evaluate(model, valid_batch)
        val_losses.append(val_loss)
        accuracies.append(accu)
        scheduler.step()  # Update the learning rate
        print(f'Epoch {epoch + 1}: Training Loss: {avg_trl:.4f}, Validation Loss: {val_loss:.4f}, Accuracy: {accu:.2f}%')


    plot_metrics(range(1, num_epochs + 1), train_losses, val_losses, accuracies)
    return train_losses, val_losses, accuracies


def evaluate(model, test_batch):
    loss_fn = nn.CrossEntropyLoss()
    model.eval()
    tl = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_batch:
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            tl += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = 100 * correct / total
    avg_tl = tl / len(test_batch)
    return avg_tl, accuracy

def plot_metrics(epochs, train_losses, val_losses, accuracies):
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Training and validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, accuracies, label='Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    train_data, validation_data, testing_data = data_handling()
    training_batch, validation_batch, testing_batch = data_load(train_data,validation_data,testing_data)
    device = torch.device('cuda' if torch.cuda. is_available() else 'cpu')
    mnist_model = ANNMODEL()
    optimizer = optim.Adam(mnist_model.parameters(), lr=0.01)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    training(20, mnist_model, training_batch, optimizer, validation_batch)
    test_loss, acc = evaluate(mnist_model, testing_batch)
    print(f'Test Loss: {test_loss:.4f}, Accuracy: {acc:.2f}%')


