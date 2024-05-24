from abc import ABC, abstractmethod
from torch.utils.data import DataLoader

from base.torch_dataset import BaseTorchDatasetForRecommendation


class Pipeline(ABC):

    def __init__(self, config, trainer, model, data_processor, device=None):
        """
        constructor for class pipeline
        :param trainer: the default trainer
        :param model:  the default model
        :param data_processor: the default data processor
        @param device: the current device that we run the pipeline
        """
        self.config = config
        self.trainer = trainer
        self.model = model
        self.data_processor = data_processor
        self.device = device

    @abstractmethod
    def process_dataset(self, dataset):
        """
        method that return processed data instances.
        :param dataset: an instance of the Dataset class
        :return: processed data instances
        """
        raise NotImplementedError("PLease implement this method")

    @abstractmethod
    def construct_dataloaders(self, data_instances, batch_size, shuffle=True, num_workers=1):
        """
        Method that constructs dataloaders using given processed data instances
        :param data_instances: the given processed data instances
        :param batch_size: number of batch size
        :param shuffle: trye if we shuffle the dataset.
        :param num_workers: number of worker used for loading the dataset
        :return:
        """
        raise NotImplementedError("PLease implement this method")

    def run(self, dataset, evaluator):
        """
        This method runs the whole pipeline including model training, selection and evaluation
        :param dataset: an instance of the dataset class
        :param evaluator: a set of evaluators
        :return: the results of the current run
        """
        raise NotImplementedError("Please implement this method")


class ToyPipeline(Pipeline):

    def process_dataset(self, dataset):
        """
        method that process the given dataset and return processed data instances
        :param dataset: an instance of the Dataset class
        :return: processed data instances.
        """
        return dataset.train_instances, dataset.valid_instances, dataset.test_instances

    def construct_dataloaders(self, data_instances, batch_size, shuffle=True, num_workers=1):
        """
        method that constructs dataloaders using given processed data instances
        :param data_instances: the processed data instances
        :param batch_size: number of batch size
        :param shuffle: True if we shuffle the data set
        :param num_workers: number of workers used for loading the dataset
        :return: a instance of torch dataloader class
        """
        torch_dataset = BaseTorchDatasetForRecommendation(
            tokenizer=self.config.tokenizer,
            instances=data_instances,
            goal2id=None,
            max_sequence_length=self.config.max_sequence_length,
            device=self.device,
            convert_example_to_feature=self.data_processor
        )

        dataloader = DataLoader(
            torch_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=torch_dataset.collate_fn,
        )
        return dataloader

    def run(self, dataset, evaluator):
        """
        This method run the whole pipeline for model training, selection and evaluation
        :param dataset: an instance of Dataset class
        :param evaluator: a set of evaluators
        :return: the results of the current run.
        """
        # process dataset
        train_instances, dev_instances, test_instances = self.process_dataset(dataset)

        # create train, dev and test dataloaders
        train_loader = self.construct_dataloaders(train_instances, batch_size=self.config.per_device_train_batch_size,
                                                  shuffle=True, num_workers=self.config.num_workers)

        dev_loader = self.construct_dataloaders(dev_instances, batch_size=self.config.per_device_eval_batch_size,
                                                shuffle=False, num_workers=self.config.num_workers)

        test_loader = self.construct_dataloaders(test_instances, batch_size=self.config.per_device_eval_batch_size,
                                                 shuffle=False, num_workers=self.config.num_workers)

        # train and select the best model
        self.trainer.train(self.model, train_loader, dev_loader, self.device)

        # evaluate the train model.
        results = evaluator(self.model, test_loader, self.device)

        # return the results of the current run
        return results
