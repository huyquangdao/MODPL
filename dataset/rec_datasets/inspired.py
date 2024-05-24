import json
import pickle
import re
import copy
import ast

import pandas as pd

from base.dataset import RecommendationDataset
# from config.constants import DURECDIAL_TARGET_GOALS


def get_movie_knowledge(movie_name, movie_db):
    row = movie_db[movie_db['title'] == movie_name]
    attributes = ['actors', 'director', 'genre', 'language']
    all_knowledge = []
    for att in attributes:
        knowledge = row[att].values
        if len(knowledge) > 10:
            knowledge = knowledge[:1]
        triples = []
        for element in knowledge:
            # only considering 1 element
            if isinstance(element, str):
                element = element.split(',')[0]
                triple = [movie_name, att, element]
                triples.append(triple)
        if triples not in all_knowledge:
            all_knowledge.extend(triples)
    return all_knowledge


def create_knowledge_base(mentioned_movies, movie_db):
    knowledge_base = []
    for movie_name in mentioned_movies:
        knowledge = get_movie_knowledge(movie_name, movie_db)
        knowledge_base.extend(knowledge)
    return knowledge_base


def process_text(text):
    text = text.replace("QUOTATION_MARK", "")
    text = text.replace("\ufffd", "\ufffd")
    text = text.strip()
    return text


def read_tsv_file(file_path):
    df = pd.read_csv(file_path, sep='\t')
    return df


def construct_conversation(temp_df):
    conv = []
    for _, row in temp_df.iterrows():
        conv.append(row['text'])
    return conv


class Inspired(RecommendationDataset):

    def __init__(self, train_data_path, dev_data_path, test_data_path, save_train_convs=True):
        # self.target_goals = DURECDIAL_TARGET_GOALS
        self.db_path = "data/INSPIRED/data/movie_database.tsv"
        super().__init__(train_data_path, dev_data_path, test_data_path, save_train_convs)
        t = 0
        for topic in self.topics:
            if topic not in self.goals:
                t += 1

    def read_data(self, data_path):
        """Function that reads the INSPIRED dataset.
        Returns:
            _type_: list of json strings
        """
        df = pd.read_csv(data_path, sep='\t')
        return df

    def repurpose_dataset(self, df):
        """convert the original goal-driven setting to the target-driven CRS setting.
        only consider recommendation-oriented conversations including food, movie, music, poi recommendation

        Args:
            data (_type_): list of json strings, each element is a conversation.

        Returns:
            _type_: list of dictionary each element corresponds to a repurposed conversation,
        """
        all_dialogue_ids = df['dialog_id'].unique()
        # group all examples by their conversation ids
        df = df.groupby('dialog_id').apply(lambda x: x)
        new_data = []

        movie_db = read_tsv_file(self.db_path)
        for id in all_dialogue_ids:
            temp_df = df[df['dialog_id'] == id]
            target_goal = None

            # convert string to dict
            temp_movie_dict = temp_df['movie_dict'].iloc[0]
            temp_movie_dict = ast.literal_eval(temp_movie_dict)

            # if there is no mentioned movie
            if len(temp_movie_dict) == 0:
                continue

            # get the last mentioned item in the movie dict
            # if's often the case that the user will accept the last mentioned item.
            try:
                temp_movie_dict = {v: k for k, v in temp_movie_dict.items()}
                target_topic = temp_movie_dict[len(temp_movie_dict) - 1]
            except:
                print(temp_movie_dict)

            check = False
            all_mentioned_movies = []
            # get target goal and topic
            for _, row in temp_df.iterrows():
                # mentioned movies ảe not nan and the current utterance is belong to the recommender.
                if len(str(row['movies'])) > 4 and row['speaker'].lower() == 'recommender':

                    # get mentioned movies in the current utterance.
                    mentioned_movies = row['movies'].split(";")
                    mentioned_movies = [x.strip() for x in mentioned_movies]
                    all_mentioned_movies.extend(mentioned_movies)

                    # if the target topic in the mentioned movies
                    if target_topic in mentioned_movies and target_topic.split('(')[0].strip().lower() in row[
                        'text'].lower():
                        target_goal = row['expert_label']
                        check = True
            # if we can not find the target.
            if not check:
                continue

            # constructing the knowledge base
            all_mentioned_movies = [x.split("(")[0].strip() for x in all_mentioned_movies]
            knowledge_base = create_knowledge_base(all_mentioned_movies, movie_db)

            target_topic = re.sub(r'\(\d+\)', '', target_topic)
            target_topic = target_topic.lower()

            new_data.append({
                "conv": temp_df,
                "knowledge": knowledge_base,
                "target_goal": target_goal,
                "target_topic": target_topic,
                "conversation": construct_conversation(temp_df),
                "user_profile": {},

                # just for convenience
                "goal_type_list": ["Greetings"] if temp_df.iloc[0]['speaker'].lower() == 'recommender' else [""]
            })

        return new_data

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

    def construct_instances(self, conv_id, conv):
        """ method that constructs input examples from a conversation
        each instance consists of task background, dialogue context and its corresponding response.

        Args:
            conv_id (_type_): the index of the input conversation
            conv (_type_): the conversation

        Returns:
            _type_: list of input instances.
        """
        # get the conversations
        instances = []
        history = [{'role': 'user', 'content': ''}]
        turn_id = 0
        temp_df = conv['conv']
        task_background = {
            "target_goal": conv['target_goal'],
            "target_topic": conv['target_topic'],
            "user_profile": conv['user_profile'],
        }
        goals = []
        topics = []
        for idx, row in temp_df.iterrows():
            if row['speaker'].lower() == 'seeker':
                history.append({'role': 'user', 'content': row['text']})

            if row['speaker'].lower() == 'recommender' and row['expert_label'] == 'no_strategy':
                history.append({'role': 'recommender', 'content': row['text']})

            if row['speaker'].lower() == 'recommender':
                # create an example here.
                if len(str(row['movies'])) > 4:
                    mentioned_movies = row['movies'].split(";")
                    mentioned_movies = [x.strip() for x in mentioned_movies]
                else:
                    mentioned_movies = []

                to_target_goal_path = []
                to_target_topic_path = []

                # goal is the sociable strategy
                goal = row['expert_label']

                # topic is the first mentioned movie
                # if these is no mentioned movie, the topic is the goal
                topic = mentioned_movies[0] if len(mentioned_movies) > 0 else goal

                # creating the reversed goal, topic paths
                tmp = len(temp_df) - 1
                while tmp > 0:
                    # get the specific turn tmp-th in the conversation.
                    tmp_row = temp_df.iloc[tmp]
                    # get the coressponding topic of that turn.
                    if len(str(tmp_row['movies'])) > 4:
                        tmp_mentioned_movies = tmp_row['movies'].split(";")
                        tmp_mentioned_movies = [x.strip() for x in tmp_mentioned_movies]
                    else:
                        tmp_mentioned_movies = []

                    # get the goal and topic
                    tmp_goal = tmp_row['expert_label']
                    tmp_topic = tmp_mentioned_movies[0] if len(tmp_mentioned_movies) > 0 else tmp_goal

                    if isinstance(tmp_topic, str):
                        tmp_topic = re.sub(r'\(\d+\)', '', tmp_topic)
                        tmp_topic = tmp_topic.lower()

                        # print(tmp_goal, tmp_topic, tmp, task_background['target_goal'], task_background['target_topic'] )

                    # if the tmp_goal is the target goal and tmp_topic is the target topic
                    if tmp_goal == task_background['target_goal'] and \
                            tmp_topic == task_background['target_topic']:
                        # we break the loop
                        # print('in here', tmp_topic)
                        # assert 1==0
                        break
                    tmp -= 1

                # loop until we meet the target goal, topic
                # increase the tmp idx
                # append the goal, topic to the paths.
                tmp_idx = turn_id
                # print(tmp, tmp_idx)
                # assert 1==0
                while tmp_idx <= tmp:
                    tmp_row = temp_df.iloc[tmp_idx]
                    # if this is the recommender turn
                    if tmp_row['speaker'].lower() == 'recommender':
                        # we get the goal and topic
                        if len(str(tmp_row['movies'])) > 4:
                            tmp_mentioned_movies = tmp_row['movies'].split(";")
                            tmp_mentioned_movies = [x.strip() for x in tmp_mentioned_movies]
                        else:
                            tmp_mentioned_movies = []

                        # get the goal and topic
                        tmp_goal = tmp_row['expert_label']
                        tmp_topic = tmp_mentioned_movies[0] if len(tmp_mentioned_movies) > 0 else tmp_goal

                        # preprocess the tmp topic
                        tmp_topic = re.sub(r'\(\d+\)', '', tmp_topic)
                        tmp_topic = tmp_topic.lower()

                        # append the goal and topic to the target_paths
                        to_target_goal_path.append(tmp_goal)
                        to_target_topic_path.append(tmp_topic)
                    tmp_idx += 1

                # append the target goal, topic to the end of the list
                # to_target_topic_path.append(task_background['target_topic'])
                # to_target_goal_path.append(task_background['target_goal'])

                if len(to_target_goal_path) == 0 and len(to_target_topic_path) == 0:
                    to_target_goal_path = [goal]
                    to_target_topic_path = [topic]

                # print(to_target_goal_path)
                # print(to_target_topic_path)
                # assert 1==0

                goal_path = copy.deepcopy(to_target_goal_path)
                topic_path = copy.deepcopy(to_target_topic_path)

                # to_target_goal_path.append(task_background['target_goal'])
                # to_target_topic_path.append(task_background['target_topic'])

                # reverse the lists.
                to_target_goal_path.reverse()
                to_target_topic_path.reverse()

                # use regex to preprocess topic
                topic = re.sub(r'\(\d+\)', '', topic)
                topic = topic.lower()

                # response is the current text
                res = process_text(row['text'])

                # get the action path to the target
                # creata the path from the next action to the target action

                # create the path from the next topic to the target topic
                # get the topic path to the target

                # default knowledge string
                knowledge = conv['knowledge'][0] if len(conv['knowledge']) > 0 else []
                # loop overall the knowledge base
                for k in conv['knowledge']:
                    h, _, t = k
                    # if head in the current response or tall in the current response
                    if h.lower() in res.lower() or t.lower() in res.lower():
                        # we assign the current knowledge to k
                        knowledge = k

                        # if the head entity in the response
                        if h.lower() in res.lower():
                            if topic == goal:
                                topic = h
                        # if the tall entity in the response
                        else:
                            if topic == goal:
                                topic = t
                        self.topics.append(topic)

                self.goals.append(goal)
                self.topics.append(topic)

                # construct an example
                example = {
                    'conv_id': conv_id,
                    'turn_id': turn_id,

                    # hittorical goal and action path
                    'pre_goals': copy.deepcopy(goals),
                    'pre_topics': copy.deepcopy(topics),
                    'dialogue_context': copy.deepcopy(history),
                    'response': res,

                    # customized knowledge base for INSPIRED.
                    'knowledge': knowledge,

                    # reversed goal and topic paths.
                    "reversed_goals": to_target_goal_path,
                    "reversed_topics": to_target_topic_path,

                    # goal path and topic path
                    "goal_path": goal_path,
                    "topic_path": topic_path,

                    # next goal and next topic
                    'goal': goal,
                    'topic': topic,

                    # goal and action path to the target.
                    "task_background": task_background,
                    # conversation:
                    "conversation": conv['conversation'],

                    # just for convenience
                    "goal_type_list": ["Greetings"] if history[0]['role'] == 'assistant' else [""]
                }

                # if topic != 'NONE':
                #     all_conversations.append(example)
                instances.append(example)
                # update the topic, goal paths
                # update the dialog history

                topics.append(topic)
                goals.append(goal)
                history.append({'role': 'assistant', 'content': res})
                turn_id += 1
        return instances
