from abc import ABC, abstractmethod
import math

from tqdm import tqdm
import numpy as np
import torch
from transformers import AdamW, get_linear_schedule_with_warmup


class Trainer(ABC):

    def __init__(self, accelerator, config, logger):
        """
        constructor for abstract class trainer
        :param accelerator: The accelerator class
        :param config: set of hyperparameters for training and developing the model
        :param logger set of loggers
        """
        self.accelerator = accelerator
        self.config = config
        self.logger = logger

    @abstractmethod
    def create_optimizer(self, model):
        """
        method that creates the optimizer for training the model
        :return: Torch optimizer
        """
        raise NotImplementedError("Please implement this method")

    def create_scheduler(self, optimizer, num_warmup_steps, max_train_steps):
        """
        method that creates the scheduler for training the model
        :return: Torch Scheduler
        """
        raise NotImplementedError("Please implement this method")

    def create_criterion(self):
        """
        method that creates the criterion for training the model
        :return: Torch Loss Function
        """
        raise NotImplementedError("Please implement this method")

    @abstractmethod
    def train_epoch(self, model, data_loader, optimizer, lr_scheduler, criterion):
        """
        method that trains the model with one epoch
        :param model: the given model
        :param data_loader: data loader for training the model
        :param optimizer: optimizer for training the model
        :param lr_scheduler:  the lr scheduler for training the model
        :param criterion: the loss function that we use to train the model
        :return: training loss
        """
        raise NotImplementedError("Please implement this method")

    @abstractmethod
    def eval_epoch(self, model, data_loader, criterion):
        """
        method that eval the model
        :param model:  the given model
        :param data_loader: the given data loader
        :param criterion: the given loss function
        :return: eval loss
        """
        raise NotImplementedError("Please implement this method")

    @abstractmethod
    def train(self, model, train_loader, dev_loader, device):
        """
        method that train the model
        :param model: the given model that we wish to train
        :param train_loader: the train data loader used to train the model
        :param dev_loader:  the dev data loader used for model selection
        :param device: the device used to train the model
        :return: the trained model
        """
        raise NotImplementedError("trainer is not implemented")


class ToyTrainer(Trainer):

    def create_criterion(self):
        """
        method that create the loss function to train the model
        :return: a torch.nn.CrossEntropyLoss object
        """
        return torch.nn.CrossEntropyLoss()

    def create_optimizer(self, model):
        """
        method that create the optimizer to train the model
        :param model: the given model that we wish to train
        :return: a torch.optim.Optimizer
        """
        modules = [model]
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for model in modules for n, p in model.named_parameters()
                           if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for model in modules for n, p in model.named_parameters()
                           if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate)
        return optimizer

    def create_scheduler(self, optimizer, num_warmup_steps, max_train_steps):
        """
        method that create the lr scheduler for training the model
        :param optimizer: the optimizer that we use to train the model
        :param num_warmup_steps: number of worm up steps
        :param max_train_steps: number of training steps.
        :return: a torch.optim.lr_scheduler
        """
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, max_train_steps)
        return lr_scheduler

    def train_epoch(self, model, data_loader, optimizer, lr_scheduler, criterion):
        """
        method that trains the model on one epoch
        :param model: the given model that we wish to train
        :param data_loader: data loader used to train the model
        :param optimizer: the optimizer used to train the model
        :param lr_scheduler:  the lr scheduler used to train the model
        :param criterion: the loss function that we use to train the model
        :return: the training loss in the current epoch
        """
        train_loss = []
        for step, batch in enumerate(data_loader):
            logits = model(batch['context'])
            loss = criterion(logits, batch['labels']) / self.config.gradient_accumulation_steps
            self.accelerator.backward(loss)
            train_loss.append(float(loss))
            # optim step
            if step % self.config.gradient_accumulation_steps == 0 or step == len(data_loader) - 1:
                if self.config.max_grad_norm is not None:
                    self.accelerator.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

        # compute average train loss
        train_loss = np.mean(train_loss) * self.config.gradient_accumulation_steps
        return train_loss

    def eval_epoch(self, model, data_loader, criterion):
        """
        method that evaluates the model on the validation set.
        :param model:  the given model
        :param data_loader:  the data loader used to evaluate the model
        :param criterion: the loss function
        :return: evaluation loss
        """
        dev_loss = []
        model.eval()
        for batch in tqdm(data_loader, disable=not self.accelerator.is_local_main_process):
            with torch.no_grad():
                logits = model(batch['context'])
                loss = criterion(logits, batch['labels'])
                dev_loss.append(float(loss))

        dev_loss = np.mean(dev_loss) * self.config.gradient_accumulation_steps
        return dev_loss

    def train(self, model, train_loader, dev_loader, device):
        """
        method that train the model
        :param model: the given that we wish to train
        :param train_loader: the train data loader
        :param dev_loader: the dev data loader
        :param device: the device we use to train the model
        :return: the trained model
        """

        # create the optimizer
        optimizer = self.create_optimizer(model)
        model, optimizer, train_dataloader = self.accelerator.prepare(model, optimizer, train_loader)

        # compute the maximum number of training steps
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.config.gradient_accumulation_steps)
        max_train_steps = self.config.num_train_epochs * num_update_steps_per_epoch

        # create the learning rate scheduler
        lr_scheduler = self.create_scheduler(optimizer, self.config.num_warmup_steps, max_train_steps)

        # create the loss function
        criterion = self.create_criterion()

        # train the model
        model.to(device)
        for epoch in range(self.config.num_train_epochs):
            model.train()
            train_loss = self.train_epoch(model, data_loader=train_loader, optimizer=optimizer,
                                          lr_scheduler=lr_scheduler,
                                          criterion=criterion)
            # dev
            dev_loss = self.eval_epoch(model, dev_loader, criterion)

        return model
