import json
import pickle
import re
import copy
from base.dataset import RecommendationDataset
from config.constants import DURECDIAL_TARGET_GOALS


class DuRecdial(RecommendationDataset):

    def __init__(self, train_data_path, dev_data_path, test_data_path, save_train_convs=True):
        self.target_goals = DURECDIAL_TARGET_GOALS
        super().__init__(train_data_path, dev_data_path, test_data_path, save_train_convs)

    def read_data(self, data_path):
        """Function that reads the Durecdial dataset.
        Returns:
            _type_: list of json strings
        """
        with open(data_path, 'r') as f:
            data = f.readlines()
            assert len(data) > 0
        return data

    def process_data(self, data):
        """method that process the conversations to get input instances.
        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        all_instances = []
        for conv_id, line in enumerate(data):
            instances = self.construct_instances(conv_id, line)
            all_instances.extend(instances)
        return all_instances

    def repurpose_dataset(self, data):
        """convert the original goal-driven setting to the target-driven CRS setting.
        only consider recommendation-oriented conversations including food, movie, music, poi recommendation

        Args:
            data (_type_): list of json strings, each element is a conversation.

        Returns:
            _type_: list of dictionary each element corresponds to a repurposed conversation,
        """
        new_data = []
        for line in data:
            line = json.loads(line)
            # each line is a conversation
            # in each conversation we have user_profile, goal_sequencs, topics_sequences and conversations.
            scenario = line['goal']
            steps = scenario.split('-->')

            # get the target goal and target topic
            i = len(steps) - 1
            while i >= 0 and ("Say goodbye" in steps[i] or 'recommendation' not in steps[i]):
                i = i - 1

            # we can not find the target recommendation goal

            if i < 0:
                continue

            # preprocessing to get the target goal and the target topic
            target_goal = re.sub(r'\(.*?\)', '', steps[i]).replace(')', '').strip()
            target_topic = steps[i].replace(target_goal, "")[1:-1].strip()

            # there are some cases such as "A. B", B is the accepted item therefore we want to get B.
            if len(target_topic.split('、')) == 2:
                target_topic = target_topic.split('、')[-1].strip()

            target_goal = re.sub(r'[0-9]', '', target_goal).replace("[]", '').strip()
            # if the target goal is not in our considered target list.
            assert target_goal in self.target_goals
            line['target_goal'] = target_goal
            line['target_topic'] = target_topic
            new_data.append(line)
        return new_data

    def construct_instances(self, conv_id, conv):
        """ method that constructs input examples from a conversation
        each instance consists of task background, dialogue context and its corresponding response.

        Args:
            conv_id (_type_): the index of the input conversation
            conv (_type_): the conversation

        Returns:
            _type_: list of input instances.
        """
        instances = []
        task_background = {
            "target_goal": conv['target_goal'],
            "target_topic": conv['target_topic'],
            "user_profile": conv['user_profile'],
        }
        utts = []
        goals = ["None"]
        topics = ["None"]
        # even for user, and odd for agent.
        role = 0
        if conv['goal_type_list'][0] == "Greetings":
            # agent starts the conversation.
            role = -1

        # # create the goal, topic paths
        # goal_path = []
        # topic_path = []

        for idx, (utt, goal, topic, knowledge) in enumerate(list(
                zip(conv['conversation'], conv['goal_type_list'], conv['goal_topic_list'], conv['knowledge']))):
            # user responses.
            self.goals.append(goal)
            self.topics.append(topic)

            if role % 2 == 0:
                utts.append({'role': 'user', 'content': utt})
            # the agent starts the conversaiton.
            elif role == -1:
                utts.append({'role': 'assistant', 'content': utt})
                goals.append(goal)
                topics.append(topic)
            # system response
            else:
                # construct the goal, topic path to the target goal, topic
                to_target_goal_path = []
                to_target_topic_path = []
                # tmp_goal = goal
                # tmp_topic = topic
                tmp_idx = idx

                # get the target goal, topic idx:
                tmp = len(conv['goal_topic_list']) - 1
                while tmp > 0 and conv['goal_type_list'][tmp] != task_background['target_goal'] and \
                        conv['goal_topic_list'][tmp] != task_background['target_topic']:
                    tmp -= 1

                # loop until we meet the target goal, topic
                # increase the tmp idx
                # append the goal, topic to the paths.
                while (tmp_idx < tmp):
                    to_target_goal_path.append(conv['goal_type_list'][tmp_idx])
                    to_target_topic_path.append(conv['goal_topic_list'][tmp_idx])
                    # tmp_goal = conv['goal_type_list'][tmp_idx]
                    # tmp_topic = conv['goal_topic_list'][tmp_idx]
                    # shift 2 due to user responses.
                    tmp_idx += 2

                # append the target goal, topic to the end of the list
                # to_target_topic_path.append(task_background['target_topic'])
                # to_target_goal_path.append(task_background['target_goal'])

                if len(to_target_goal_path) == 0 and len(to_target_topic_path) == 0:
                    to_target_goal_path = [goal]
                    to_target_topic_path = [topic]

                goal_path = copy.deepcopy(to_target_goal_path)
                topic_path = copy.deepcopy(to_target_topic_path)

                # to_target_goal_path.append(task_background['target_goal'])
                # to_target_topic_path.append(task_background['target_topic'])

                # reverse the lists.
                to_target_goal_path.reverse()
                to_target_topic_path.reverse()

                # constructing an instance.
                instance = {
                    "conv_id": conv_id,
                    "response": utt,
                    "goal": goal,
                    "topic": topic,
                    "reversed_goals": to_target_goal_path,
                    "reversed_topics": to_target_topic_path,
                    "knowledge": knowledge,
                    "pre_goals": copy.deepcopy(goals),
                    "pre_topics": copy.deepcopy(topics),
                    "dialogue_context": copy.deepcopy(utts),
                    "task_background": copy.deepcopy(task_background),
                    "goal_path": goal_path,
                    "topic_path": topic_path
                }
                instances.append(instance)
                utts.append({'role': 'assistant', 'content': utt})
                goals.append(goal)
                topics.append(topic)
            role = role + 1

        # # assign goal, topic paths to instances
        # for instance in instances:
        #     instance['goal_path'] = goals
        #     instance['topic_path'] = topics
        return instances
