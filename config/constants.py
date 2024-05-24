USER_TOKEN = "[USER]"
SYSTEM_TOKEN = "[SYSTEM]"
KNOW_TOKEN = "[KNOW]"
PATH_TOKEN = "[PATH]"
SEP_TOKEN = "[SEP]"
PROFILE_TOKEN = "[PROFILE]"
CONTEXT_TOKEN = "[CONTEXT]"
GOAL_TOKEN = "[GOAL]"
TARGET = "[TARGET]"
TOPIC_TOKEN = "[TOPIC]"
PAD_TOKEN = "<pad>"
IGNORE_INDEX = -100

special_tokens_dict = {
    'additional_special_tokens': [USER_TOKEN, SYSTEM_TOKEN, KNOW_TOKEN, PATH_TOKEN, SEP_TOKEN, PROFILE_TOKEN,
                                  CONTEXT_TOKEN, GOAL_TOKEN, TARGET],
}

DURECDIAL_TARGET_GOALS = [
    "Movie recommendation",
    "Food recommendation",
    "Music recommendation",
    "POI recommendation",
]

DURECDIALGOALS = {
    'Ask about weather',
    'Play music',
    'Q&A',
    'Music on demand',
    'Movie recommendation',
    'Chat about stars',
    'Say goodbye',
    'Music recommendation',
    'Ask about date',
    'Ask questions',
    'Greetings',
    'POI recommendation',
    'Food recommendation',
}
