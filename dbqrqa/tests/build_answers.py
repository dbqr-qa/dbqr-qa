import argparse
import json
from os.path import join

from dbqrqa.dataset import TableDataset

DEFAULT_DATA_PATH = 'data'


def main(args: argparse.Namespace) -> None:
    """
    Add noise to the practice labels for testing.
    """

    dataset = TableDataset()
    practice = dataset.practice

    with open(join(args.data, 'tests', 'evaluation', 'noise.json')) as file:
        noise = json.load(file)

    answers = {}

    for chat_id, chat in practice.labels.items():
        answers[chat_id] = {}

        for question_id, label in chat.items():
            answers[chat_id][question_id] = label

            if chat_id in noise and question_id in noise[chat_id]:
                answers[chat_id][question_id] = noise[chat_id][question_id]

    with open(join(args.data, 'tests', 'evaluation', 'answers.json'), 'w') as file:
        json.dump(answers, file, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=DEFAULT_DATA_PATH)
    args = parser.parse_args()
    main(args)
