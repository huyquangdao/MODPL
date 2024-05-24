from __future__ import print_function
from collections import defaultdict

import torch

from torch.utils.data import Dataset as TorchDataset
from torch.nn.utils.rnn import pad_sequence
from config.constants import IGNORE_INDEX

# IGNORE_INDEX = -100

"""
Reuse the code from MMCTS source code.
"""


class BaseTorchDatasetForRecommendation(TorchDataset):

    def __init__(self, tokenizer, instances, goal2id=None, max_sequence_length=512, padding='max_length',
                 pad_to_multiple_of=True, device=None, convert_example_to_feature=None, max_target_length=50,
                 is_test=False, is_gen=False):
        """
        constructor for the BaseTorchDataset Class
        @param tokenizer: an huggingface tokenizer
        @param instances: a list of instances
        @param goal2id: a dictionary which maps goal to index.
        @param max_sequence_length: the maximum length of the input sequence.
        @param padding: type of padding
        @param pad_to_multiple_of: pad to multiple instances
        @param device: device to allocate the data, eg: cpu or gpu
        @param convert_example_to_feature: a function that convert raw instances to
        corresponding inputs and labels for the model.
        @param max_target_length the maximum number of the target sequence (response generation only)
        @param is_test True if inference step False if training step
        @param is_gen True if response generation else False
        """
        super(BaseTorchDatasetForRecommendation, self).__init__()
        self.max_sequence_length = max_sequence_length
        self.tokenizer = tokenizer
        self.goal2id = goal2id
        self.pad_to_multiple_of = pad_to_multiple_of
        self.padding = padding
        self.device = device
        self.max_target_length = max_target_length
        self.is_test = is_test
        self.is_gen = is_gen
        self.instances = self.preprocess_data(instances, convert_example_to_feature)

    def __len__(self):
        """
        method that returns the number of instances in the dataset.
        @return: an integer which is the number of instances in the training dataset.
        """
        return len(self.instances)

    def __getitem__(self, idx):
        """
        function that return an instance from the set of all instances.
        @param idx: the index of the returned instances.
        @return: an instance.
        """
        instance = self.instances[idx]
        return instance

    def collate_fn(self, batch):

        input_features = defaultdict(list)
        labels = []
        for instance in batch:
            input_features['input_ids'].append(instance['input_ids'])
            labels.append(instance['label'])

        # padding the input features
        input_features = self.tokenizer.pad(
            input_features, padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.max_sequence_length
        )
        # convert features to torch tensors
        for k, v in input_features.items():
            if not isinstance(v, torch.Tensor):
                input_features[k] = torch.as_tensor(v, device=self.device)

        # labels for response generation task
        if self.is_gen:
            labels = pad_sequence(
                [torch.tensor(label, dtype=torch.long) for label in labels],
                batch_first=True, padding_value=IGNORE_INDEX)
            labels = labels.to(self.device)
        # labels for goal prediction task
        else:
            labels = torch.LongTensor(labels).to(self.device)

        new_batch = {
            "context": input_features,
            "labels": labels
        }
        return new_batch

    def preprocess_data(self, instances, convert_example_to_feature):
        """
        method that preprocess an data instances
        @param instances: an instance from the data
        @return: a processed instance which consists of input_ids and labels
        """
        processed_instances = []
        for instance in instances:
            if not self.is_gen:
                input_ids, label = convert_example_to_feature(self.tokenizer, instance, self.max_sequence_length,
                                                              self.goal2id)
            else:
                input_ids, label = convert_example_to_feature(self.tokenizer, instance, self.max_sequence_length,
                                                              self.max_target_length, self.is_test)
            new_instance = {
                "input_ids": input_ids,
                "label": label
            }
            processed_instances.append(new_instance)
        return processed_instances


class BaseTorchDatasetForNegotiation(TorchDataset):

    def __init__(self):
        super(BaseTorchDatasetForNegotiation, self).__init__()

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass


class BaseTorchDatasetForClarification(TorchDataset):

    def __init__(self):
        super(BaseTorchDatasetForClarification, self).__init__()

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass
