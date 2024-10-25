import json
import pickle
from os import listdir
from os.path import isdir, join
from typing import Any, Dict, Union

import pandas as pd

from dbqrqa.evaluation import (
    DEFAULT_EVALUATOR,
    DEFAULT_GPT_MODEL,
    DEFAULT_RETRY,
    OUTPUT_TYPES, 
    evaluate)
from dbqrqa.types import ANSWER_TYPES

STAGES = ('practice', 'train', 'test')

DEFAULT_DATA_PATH = 'data'


class TableSplit:
    """
    A split of the dataset for the practicing, training, or testing stage.
    """

    def __init__(
        self, 
        stage: str, 
        data_path: str = DEFAULT_DATA_PATH):

        self.stage = stage
        self.data_path = data_path
        self.answers: Union[ANSWER_TYPES, None] = None

        self.load()

    def load(self):
        stage_path = join(self.data_path, self.stage)
        question_path = join(stage_path, 'questions')
        table_path = join(stage_path, 'tables')

        self.chats = {}

        for chat_id in listdir(question_path):
            chat = {}

            for question_id in range(1, 11):
                file_path = join(
                    question_path, 
                    chat_id, 
                    'question-%02d.json' % question_id)
                
                with open(file_path) as file:
                    sample = json.load(file)

                file_path = join(
                    table_path,
                    chat_id,
                    'question-%02d.pkl' % question_id)
                
                with open(file_path, 'rb') as file:
                    sample['tables'] = pickle.load(file)
                
                chat[str(question_id)] = sample
        
            self.chats[chat_id] = chat

    def answer(
        self, 
        chat_id: str, 
        question_id: Union[int, str], 
        answer: ANSWER_TYPES):

        # Check chat ID
        chat = chat_id.split('-')

        assert len(chat) == 3, 'Invalid chat ID'
        assert chat[0] == 'chat', 'Invalid chat ID'
        assert chat[1].isnumeric() and len(chat[1]) == 1, 'Invalid chat ID'
        assert chat[2].isnumeric() and len(chat[2]) == 2, 'Invalid chat ID'

        # Check question ID
        question_id = str(question_id)

        assert question_id.isnumeric(), 'Question ID must be an integer'

        # Remove preceeding zeros
        question_id = str(int(question_id))

        if int(question_id) < 1 or int(question_id) > 10:
            raise ValueError('Question ID must be between 1 and 10.')

        if self.answers is None:
            self.answers = {}

        if chat_id not in self.answers:
            self.answers[chat_id] = {}
        
        self.answers[chat_id][question_id] = answer

    def evaluate(
        self, 
        evaluator: str = DEFAULT_EVALUATOR,
        model: str = DEFAULT_GPT_MODEL,
        retry: int = DEFAULT_RETRY,
        openai_key: Union[str, None] = None,
        backup_path: Union[str, None] = None,
        is_notebook: bool = False) -> OUTPUT_TYPES:

        if self.answers is None:
            raise RuntimeError('No answers assigned to this dataset.' \
                'See README file to learn how to assign the answers.')

        return evaluate(self.questions, self.answers, self.labels,
            evaluator=evaluator, model=model, retry=retry,
            openai_key=openai_key, backup_path=backup_path,
            is_notebook=is_notebook)

    def _get_property(self, key: str) -> Dict[str, Dict[str, Any]]:
        chats = {}

        for chat_id, chat in self.chats.items():
            chats[chat_id] = {}

            for question_id, sample in chat.items():
                chats[chat_id][question_id] = sample[key]
        
        return chats

    @property
    def questions(self) -> Dict[str, Dict[str, str]]:
        return self._get_property('question')
    
    @property
    def labels(self) -> ANSWER_TYPES:
        return self._get_property('answer')
    
    @property
    def queries(self) -> Dict[str, Dict[str, str]]:
        return self._get_property('queries')
    
    @property
    def tables(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        return self._get_property('tables')


class TableDataset:
    """
    A dataset class for the project's shared task where table inputs are given
    (no database query required). For more information, visit https://dbqr-qa.github.io
    """

    def __init__(self, data_path: str = DEFAULT_DATA_PATH):
        self.data_path = data_path
        self.practice: Union[TableSplit, None] = None
        self.train: Union[TableSplit, None] = None
        self.test: Union[TableSplit, None] = None
        self.load()

    def load(self):
        for stage in STAGES:
            if isdir(join(self.data_path, stage)):
                setattr(self, stage, TableSplit(stage, self.data_path))
