import json
import pickle
from os import listdir
from os.path import isdir, join
from typing import Any, Dict

import pandas as pd

from dbqrqa.types import ANSWER_TYPES

STAGES = ('practice', 'train', 'test')

DEFAULT_DATA_PATH = 'data'


class TableSplit:
    def __init__(
        self, 
        stage: str, 
        data_path: str = DEFAULT_DATA_PATH):

        self.stage = stage
        self.data_path = data_path

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
    practice: TableSplit
    train: TableSplit
    test: TableSplit

    def __init__(self, data_path: str = DEFAULT_DATA_PATH):
        self.data_path = data_path
        self.load()

    def load(self):
        for stage in STAGES:
            if isdir(join(self.data_path, stage)):
                setattr(self, stage, TableSplit(stage, self.data_path))
