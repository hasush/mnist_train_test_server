import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from mnist_classifier.mnist_dataset import MnistDataset
from mnist_classifier.model import Model
from mnist_classifier.config import Config

class TrainTest:

    def __init__(self):

        self.config = Config()

        # Set the cpu or the gpu if it is available.
        if self.config.gpu:
            if torch.cuda.is_available():
                self.torch_device=torch.device('cuda')
            else: 
                raise ValueError("GPU is not available. Configure GPU or set config.gpu to False")
        else:
            self.torch_device=torch.device('cpu')

        # Get the model and set the optimizer, learning rate scheduler, and 
        self.model = Model()
        self.optimizer = optim.Adadelta(self.model.parameters())
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 
                                                   step_size=self.config.lr_decay_step_size, 
                                                   gamma=self.config.lr_gamma)
        self.loss=torch.nn.NLLLoss()
        self.train_data_loader=None
        self.val_data_loader=None
        self.epoch=None


    def _train_step(self, batch_idx, data, target):

        # Put the data and targets to the device, zero out gradients, and run
        # the data through the model.
        data, target = data.to(self.torch_device), target.to(self.torch_device)
        self.optimizer.zero_grad()
        output = self.model(data)

        # Compute loss, backpropagate it, and decay optimizer parameters.
        loss = self.loss(output, target)
        loss.backward()
        self.optimizer.step()

        # Info.
        if batch_idx % 128 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                self.epoch, batch_idx * len(data), len(self.train_data_loader.dataset),
                100. * batch_idx / len(self.train_data_loader), loss.item()))


    def _evaluate_step(self, evaluate_loader):

        # Set model to evaluation.
        self.model.eval()

        # Convert the input to a dataloader if it is of type dataset.
        if type(evaluate_loader) == type(torch.utils.data.DataLoader(torch.utils.data.Dataset())):
            data_loader = evaluate_loader
        else:
            raise ValueError("Input to evaluate step must be of type torch.utils.data.DataLoader")

        # No accumulation of gradient for evaluation.
        loss = 0
        correct = 0
        with torch.no_grad():

            # Loop through the dataset.
            for data, target in data_loader:

                # Run batch size of data through network.
                data, target = data.to(self.torch_device), target.to(self.torch_device)
                output = self.model(data)

                # Compute loss across the batch..
                loss += F.nll_loss(output, target, reduction='sum').item() 

                # The prediction is given by the highest logit value.
                prediction = output.argmax(dim=1, keepdim=True)
                correct += prediction.eq(target.view_as(prediction)).sum().item()

        # Normalize by the size of the dataset.
        loss /= len(data_loader.dataset)

        # Info.
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            loss, correct, len(data_loader.dataset),
            100. * correct / len(data_loader.dataset)))


    def launch_training(self, train_dataset, val_dataset):

        # Convert data sets to a data loader.
        self.train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.batch_size)
        self.val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.batch_size)
        num_train_batches=len(self.train_data_loader)
        batches_til_val=self.config.fraction_of_epoch_before_val*num_train_batches

        # Set model to training and starting looping over the dataset.
        self.model.to(self.torch_device)
        self.model.train()

        for self.epoch in range(1, self.config.epochs+1):

            # Get the current batch for the epoch.
            for batch_idx, (data, target) in enumerate(self.train_data_loader):

                # Run train step.
                self._train_step(batch_idx, data, target)

                # Run validation data if it is time to do so.
                if (batch_idx+1) % batches_til_val==0:
                    self._evaluate_step(self.val_data_loader)
                    self.model.train()
                    torch.save(self.model.state_dict(), self.config.model_training_checkpoint_path)

            # Decay the learning rate parameters.
            self.scheduler.step()

    
    def launch_evaluation(self, test_dataset):

        # Convert data sets to a data loader.
        self.test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.config.batch_size)
        
        # Load model state dict.
        if self.torch_device=='cuda':
            self.model.load_state_dict(torch.load(self.config.model_evaluate_checkpoint_path))
        else:
            self.model.load_state_dict(torch.load(self.config.model_evaluate_checkpoint_path, map_location='cpu'))

        self.model.to(self.torch_device)

        # Perform evaluation.
        self._evaluate_step(self.test_data_loader)

  



  
                
 
