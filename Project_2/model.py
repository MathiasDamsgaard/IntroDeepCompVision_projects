import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torch.nn as nn
import torch

class VideoClassifier(nn.Module):
    def __init__(self, n_classes=10):
        super(VideoClassifier, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Transfer learning with pretrained ResNet18
        self.base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, n_classes)
        self.base_model.to(self.device)

        # freeze backbone
        for name, param in self.base_model.named_parameters():
            if not name.startswith('fc'):
                param.requires_grad = False

        # optimizer on trainable params only
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                                 self.base_model.parameters()),
                                                 lr=1e-3, weight_decay=1e-2)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, num_epochs, train_loader, test_loader):
        train_acc, test_acc = [], []
        self.base_model.to(self.device)

        for _ in range(num_epochs):
            # For each epoch
            self.base_model.train()
            train_correct = 0

            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                # Zero the gradients computed for each weight
                self.optimizer.zero_grad()
                
                # Forward pass your image through the network
                output = self.base_model(data)
                
                # Compute the loss (compute log_softmax over class dimension)
                loss = self.criterion(output, target)
                
                # Backward pass through the network
                loss.backward()
                
                # Update the weights
                self.optimizer.step()
                
                # Compute how many were correctly classified
                predicted = output.argmax(1)
                train_correct += (target == predicted).sum().cpu().item()

            # Compute the test accuracy
            self.base_model.eval()
            test_correct = 0
            
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                with torch.no_grad():
                    output = self.base_model(data)
                predicted = output.argmax(1)
                test_correct += (target == predicted).sum().cpu().item()

            train_acc.append(train_correct/len(train_loader.dataset))
            test_acc.append(test_correct/len(test_loader.dataset))
            print("Accuracy train: {train:.1f}%\t val: {test:.1f}%".format(train=100*train_acc[-1], test=100*test_acc[-1]))

        return train_acc, test_acc
    
    def evaluate(self, data):
        self.base_model.eval()
        data = data.to(self.device)
        with torch.no_grad():
            output = self.base_model(data)
        predictions = output.cpu()
        return predictions