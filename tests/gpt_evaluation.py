import argparse
import json
from os.path import join

from dbqrqa.dataset import TableDataset
from dbqrqa.evaluation import evaluate

DEFAULT_OPENAI_KEY_PATH = join('data', 'keys', 'openai.txt')
DEFAULT_EVALUATOR = 'gpt-binary'
DEFAULT_MODEL = 'gpt-4o'
DEFAULT_RETRY = 3
DEFAULT_BACKUP_PATH = join('data', 'backup')
DEFAULT_DATA_PATH = 'data'


def main(args: argparse.Namespace) -> None:
    """
    The GPT evaluator is not part of the standard unit test
    due to the cost and security considerations.

    Call this special test function with an API key to test
    the evaluator.
    """

    dataset = TableDataset()
    practice = dataset.practice

    with open(args.key) as file:
        api_key = file.read().strip()

    with open(join(args.data, 'tests', 'evaluation', 'contaminate.json')) as file:
        changes = json.load(file)

    answers = {}

    for chat_id, chat in practice.labels.items():
        answers[chat_id] = {}

        for question_id, label in chat.items():
            answers[chat_id][question_id] = label

            if chat_id in changes and question_id in changes[chat_id]:
                answers[chat_id][question_id] = changes[chat_id][question_id]

    accuracy, _ = evaluate(practice.questions, answers, practice.labels,
        args.evaluator, args.model, args.retry, api_key, args.backup)
    
    print('Accuracy:', accuracy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluator', type=str, default=DEFAULT_EVALUATOR,
        help='Evaluation method', choices=('gpt-binary', 'gpt-score'))
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
        help='OpenAI model')
    parser.add_argument('--retry', type=int, default=DEFAULT_RETRY,
        help='Maximum number of retries for invalid outputs')
    parser.add_argument('--key', type=str, default=DEFAULT_OPENAI_KEY_PATH,
        help='Path to OpenAI API key')
    parser.add_argument('--backup', type=str, default=DEFAULT_BACKUP_PATH,
        help='Backup path for the model\'s outputs and scores')
    parser.add_argument('--data', type=str, default=DEFAULT_DATA_PATH)
    args = parser.parse_args()
    main(args)
