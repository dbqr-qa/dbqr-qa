import json
import unittest
from os import listdir
from os.path import join

from dbqrqa.evaluation import evaluate

QUESTION_PATH = join('data', 'practice', 'questions')
TEST_PATH = join('data', 'tests', 'evaluation')


class EvaluationTestCase(unittest.TestCase):
    def build_test_input(self):
        self.questions = {}
        self.answers = {}
        self.labels = {}

        for chat_id in listdir(QUESTION_PATH):
            questions, labels, answers = {}, {}, {}

            for question_id in range(1, 11):
                chat_file = join(
                    QUESTION_PATH, 
                    chat_id, 
                    'question-%02d.json' % question_id)
            
                with open(chat_file) as file:
                    data = json.load(file)
                
                questions[str(question_id)] = data['question']
                answers[str(question_id)] = data['answer']
                labels[str(question_id)] = data['answer']
            
            self.questions[chat_id] = questions
            self.answers[chat_id] = answers
            self.labels[chat_id] = labels

    def test_heuristic_basic(self):
        self.build_test_input()

        accuracy, _ = evaluate(
            self.questions, self.answers, self.labels)
        
        self.assertEqual(accuracy, 1.0)

    def test_heuristic_contaminate(self):
        self.build_test_input()

        with open(join(TEST_PATH, 'contaminate.json')) as file:
            changes = json.load(file)
        
        for chat_id, items in changes.items():
            for question_id, answer in items.items():
                self.answers[chat_id][question_id] = answer

        accuracy, _ = evaluate(
            self.questions, self.answers, self.labels)
        
        self.assertEqual(accuracy, .82)


if __name__ == "__main__":
    unittest.main()
