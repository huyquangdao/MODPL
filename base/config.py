from abc import ABC, abstractmethod
import yaml


class Config(ABC):

    def __init__(self, params):
        """
        constructor for class Config
        :param params: parameters for training, developing and running the model
        """
        self.params = params

    def set_params(self, params):
        """
        method that sets parameters for class Config.
        :param params: a dictionary where keys are parameter names and values are the parameter values
        :return: None
        """
        for k, v in params.items():
            assert k in self.params.keys()
            self.params[k] = v

    def get_params(self):
        """
        Method that returns the parameters for class Config
        :return: a dictionary
        """
        return self.params

    def load_config_from_yaml_file(self, file_path):
        """
        Method that loads the parameters from a yaml configuration file
        :param file_path: the path to the config file
        :return: None
        """
        with open(file_path, 'r') as f:
            self.params = yaml.load(f, loader=yaml.FullLoader)
