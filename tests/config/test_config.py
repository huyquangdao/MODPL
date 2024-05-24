from unittest import TestCase

from base.config import Config


class TestConfig(TestCase):
    def test_set_params(self):
        self.fail()

    def test_get_params(self):
        self.fail()

    def test_load_config_from_yaml_file(self):
        yaml_file_path = "config/datasets/durecdial.yaml"
        config = Config({}).load_config_from_yaml_file(yaml_file_path)
        print(config)
