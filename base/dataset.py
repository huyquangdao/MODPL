from abc import ABC, abstractmethod
from collections import defaultdict


class Dataset(ABC):

    def __init__(self, train_data_path, dev_data_path, test_data_path, save_train_convs=True):
        """
        constructor for the abstract class Dataset
        :param train_data_path:
        :param dev_data_path:
        :param test_data_path:
        :param save_train_convs:
        """
        self.train_data_path = train_data_path
        self.dev_data_path = dev_data_path
        self.test_data_path = test_data_path
        self.save_train_convs = save_train_convs

    def pipeline(self, data_path):
        """method that employs that data pipeline including read_data, repurpose_data and progress_data
        """
        data = self.read_data(data_path=data_path)
        data = self.repurpose_dataset(data)
        if self.save_train_convs and 'train' in data_path:
            self.train_convs = data
        data = self.process_data(data)
        return data

    @abstractmethod
    def construct_instances(self, conv_id, conv):
        """
        method that converts a conversation to a list of inputs and their corresponding outputs.
        @param conv_id: the index of the conversation
        @param conv: the conversation
        @return: a list of instances
        """
        raise NotImplementedError()

    @abstractmethod
    def read_data(self, data_path):
        """function that reads the data from input file

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()

    @abstractmethod
    def process_data(self, data):
        """Function that process the data given the read data.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()

    @abstractmethod
    def repurpose_dataset(self, data):
        """Function that convert the original dataset from goal-driven setting to target-driven setting.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()

    @abstractmethod
    def return_infor(self):
        """
        method that return information regarding the dataset
        :return: a dictionary that contains the information of the dataset
        """
        raise NotImplementedError()


class RecommendationDataset(Dataset):

    def __init__(self, train_data_path, dev_data_path, test_data_path, save_train_convs=True):
        super().__init__(train_data_path, dev_data_path, test_data_path, save_train_convs)

        self.topics = []
        self.goals = []
        self.save_train_convs = save_train_convs
        self.train_convs = None

        self.train_instances = self.pipeline(self.train_data_path)
        self.dev_instances = self.pipeline(self.dev_data_path)
        self.test_instances = self.pipeline(self.test_data_path)

        self.log_goal = True
        if self.log_goal:
            goal_dict = defaultdict(int)
            # log goal count
            for goal in self.goals:
                goal_dict[goal] += 1

            # log target item w.r.t data split
            train_target_items = []
            dev_target_items = []
            test_target_items = []

            # log target w.r.t different domains
            movie_target_items = defaultdict(list)
            music_target_items = defaultdict(list)
            food_target_items = defaultdict(list)
            poi_target_items = defaultdict(list)

            for instance in self.train_instances:
                train_target_items.append(instance['task_background']['target_topic'])

                if instance['task_background']['target_goal'] == 'Movie recommendation':
                    movie_target_items['train'].append(instance['task_background']['target_topic'])
                if instance['task_background']['target_goal'] == 'Music recommendation':
                    music_target_items['train'].append(instance['task_background']['target_topic'])
                if instance['task_background']['target_goal'] == 'Food recommendation':
                    food_target_items['train'].append(instance['task_background']['target_topic'])

                if instance['task_background']['target_goal'] == 'POI recommendation':
                    poi_target_items['train'].append(instance['task_background']['target_topic'])

            for instance in self.dev_instances:
                dev_target_items.append(instance['task_background']['target_topic'])

                if instance['task_background']['target_goal'] == 'Movie recommendation':
                    movie_target_items['dev'].append(instance['task_background']['target_topic'])
                if instance['task_background']['target_goal'] == 'Music recommendation':
                    music_target_items['dev'].append(instance['task_background']['target_topic'])
                if instance['task_background']['target_goal'] == 'Food recommendation':
                    food_target_items['dev'].append(instance['task_background']['target_topic'])

                if instance['task_background']['target_goal'] == 'POI recommendation':
                    poi_target_items['dev'].append(instance['task_background']['target_topic'])

            for instance in self.test_instances:
                test_target_items.append(instance['task_background']['target_topic'])

                if instance['task_background']['target_goal'] == 'Movie recommendation':
                    movie_target_items['test'].append(instance['task_background']['target_topic'])
                if instance['task_background']['target_goal'] == 'Music recommendation':
                    music_target_items['test'].append(instance['task_background']['target_topic'])
                if instance['task_background']['target_goal'] == 'Food recommendation':
                    food_target_items['test'].append(instance['task_background']['target_topic'])

                if instance['task_background']['target_goal'] == 'POI recommendation':
                    poi_target_items['test'].append(instance['task_background']['target_topic'])

            print(
                f"Statistics by data splits: Train: {len(list(set(train_target_items)))}, Dev: {len(list(set(dev_target_items)))}, Test: {len(list(set(test_target_items)))}")

            for t in ['train', 'dev', 'test']:
                print(
                    f"Statistics by domain splits {t}: Movie: {len(list(set(movie_target_items[t])))}, Music: {len(list(set(music_target_items[t])))}, Food: {len(list(set(food_target_items[t])))}, POI: {len(list(set(poi_target_items[t])))}")

        self.goals = list(set(self.goals))
        self.topics = list(set(self.topics))

    def return_infor(self):
        """function that returns information about the dataset

        Returns:
            _type_: dictionary
        """
        infor_dict = {
            "num_topics": len(self.topics),
            "num_goals": len(self.goals),
            "train_instances": len(self.train_instances),
            "dev_instances": len(self.dev_instances),
            "test_instances": len(self.test_instances)

        }
        return infor_dict


class NegotiationDataset(Dataset):

    def __init__(self, train_data_path, dev_data_path, test_data_path, save_train_convs=True):
        super().__init__(train_data_path, dev_data_path, test_data_path, save_train_convs)

    def return_infor(self):
        pass


class ClarificationDataset(Dataset):

    def __init__(self, train_data_path, dev_data_path, test_data_path, save_train_convs=True):
        super().__init__(train_data_path, dev_data_path, test_data_path, save_train_convs)

    def return_infor(self):
        pass
