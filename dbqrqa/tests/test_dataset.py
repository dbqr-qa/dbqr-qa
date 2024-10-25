import json
import unittest
from os.path import join

from dbqrqa.dataset import TableDataset

TEST_PATH = join('data', 'tests', 'evaluation')


class TableDatasetTestCase(unittest.TestCase):
    def test_dataset(self):
        dataset = TableDataset()

        self.assertGreater(len(dataset.practice.questions), 0)
        self.assertGreater(len(dataset.practice.labels), 0)
        self.assertGreater(len(dataset.practice.queries), 0)
        self.assertGreater(len(dataset.practice.tables), 0)

        with open(join(TEST_PATH, 'answers.json')) as file:
            dataset.practice.answers = json.load(file)

        dataset.practice.answer('chat-1-01', 1, 'lower')
        
        accuracy, _ = dataset.practice.evaluate()
        self.assertEqual(accuracy, 0.82)


if __name__ == "__main__":
    unittest.main()
