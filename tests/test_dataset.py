import unittest

from dbqrqa.dataset import TableDataset


class TableDatasetTestCase(unittest.TestCase):
    def test_dataset(self):
        dataset = TableDataset()

        self.assertGreater(len(dataset.practice.questions), 0)
        self.assertGreater(len(dataset.practice.answers), 0)
        self.assertGreater(len(dataset.practice.queries), 0)
        self.assertGreater(len(dataset.practice.tables), 0)


if __name__ == "__main__":
    unittest.main()
