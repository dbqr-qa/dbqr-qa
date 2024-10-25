import json
import logging
from typing import Dict, Iterable, List, Tuple, Union
from os.path import isfile, join
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from dbqrqa.prompts import (
    GPT_PROMPT_SYSTEM,
    GPT_PROMPT_BINARY, 
    GPT_PROMPT_SCORE)
from dbqrqa.types import (
    VALUE_TYPES,
    ANSWER_TYPES)

OUTPUT_TYPES = Tuple[float, Dict[str, Dict[str, float]]]

EVALUATORS = ('heuristic', 'gpt-binary', 'gpt-score')

DEFAULT_EVALUATOR = 'heuristic'
DEFAULT_GPT_MODEL = 'gpt-4o'
DEFAULT_RETRY = 3


def _match_heuristic(answer: VALUE_TYPES, label: VALUE_TYPES) -> bool:
    """
    Determine the label type (string, int, or float) and convert
    both the answer and label to the same format, then compare.
    """

    if isinstance(label, str):
        return str(answer).lower().strip() == label.lower().strip()

    elif isinstance(label, int):
        try:
            return int(answer) == label

        except (ValueError, TypeError):
            return False
    
    elif isinstance(label, float):
        try:
            return '%.2f' % float(answer) == '%.2f' % label

        except (ValueError, TypeError):
            return False


def evaluate_heuristic(answer: ANSWER_TYPES, label: ANSWER_TYPES) -> int:
    """
    Evaluate a single answer-label pair using the heuristic evaluator.
    For more information, visit https://dbqr-qa.github.io/quickstart.html#heuristic
    """

    if isinstance(answer, set):
        answer = list
    
    if isinstance(label, list):
        if not isinstance(answer, list) or len(answer) != len(label):
            return 0
        
        answer = sorted(answer)
        label = sorted(label)

        for item, ref in zip(answer, label):
            if not _match_heuristic(item, ref):
                return 0

        return 1

    elif isinstance(label, dict):
        if not isinstance(answer, dict) or len(answer) != len(label):
            return 0
        
        for key, ref in label.items():
            if key not in answer:
                return 0

            if not _match_heuristic(answer[key], ref):
                return 0
        
        return 1

    else:
        return int(_match_heuristic(answer, label))


def init_gpt_messages() -> List[Dict[str, str]]:
    """
    Initialize GPT's input by explaning the task.
    """
    return [{'role': 'system', 'content': GPT_PROMPT_SYSTEM}]


def evaulate_gpt(
    question: str,
    answer: ANSWER_TYPES,
    label: ANSWER_TYPES,
    messages: List[Dict[str, str]],
    prompt: str,
    client: OpenAI,
    model: str,
    scoring: str = 'binary',
    retry: int = DEFAULT_RETRY) -> Tuple[float, Dict[str, str]]:
    
    """
    Evaluate a single answer-label pair using the GPT evaluator.
    For more information, visit https://dbqr-qa.github.io/quickstart.html#gpt

    Use init_gpt_messages() to initialize `messages` (GPT's input).
    This function will automatically append the GPT's response to `messages` before
    returning the score and the updated `messages`.

    For binary evaluator, the score can either be zero or one.
    For scoring evaluator, the score ranges from zero to one on 0.1 intervals.

    The `retry` parameter sets the limit of how many times the function
    will try to get the score if the previous attempts failed due to
    invalid GPT's responses.
    """

    def append(messages: List[Dict[str, str]], content: str):
        messages.append({
            'role': 'assistant',
            'content': content})
        
        return messages
    
    if scoring not in ('binary', 'score'):
        raise KeyError('Invalid scoring choice. ' \
            'The valid choices are binary or score.')

    content = prompt.replace('{{question}}', question) \
                .replace('{{answer}}', str(answer)) \
                .replace('{{label}}', str(label))

    messages.append({
        'role': 'user',
        'content': content})

    attempt = 0

    while attempt < retry:
        response = client.chat.completions.create(
            model=model,
            messages=messages)
        
        content = response.choices[0].message.content

        # Extract the JSON output by detecting lines in a code block
        is_json_lines = False
        json_output = ''

        for line in content.split('\n'):
            if '```' in line:
                is_json_lines = not is_json_lines
            
            elif is_json_lines:
                json_output += line + '\n'
        
        try:
            output = json.loads(json_output)

            if scoring == 'binary':
                if output['result'] == 'yes':
                    return 1, append(messages, content)
                
                elif output['result'] == 'no':
                    return 0, append(messages, content)

                else:
                    logging.warning('Invalid GPT output. Retrying.')
                    attempt += 1

            else:
                score = int(output['result'])

                if 0 <= score <= 10:
                    return float(score) / 10, append(messages, content)

                else:
                    logging.warning('Invalid GPT output. Retrying.')
                    attempt += 1
        
        except (json.decoder.JSONDecodeError, KeyError, ValueError):
            logging.warning('Invalid GPT output. Retrying.')
            attempt += 1

    raise RuntimeError('Invalid GPT output exceeded retry limit.')


def evaluate(
    questions: Dict[str, Dict[str, str]],
    answers: ANSWER_TYPES,
    labels: ANSWER_TYPES,
    evaluator: str = DEFAULT_EVALUATOR,
    model: str = DEFAULT_GPT_MODEL,
    retry: int = DEFAULT_RETRY,
    openai_key: Union[str, None] = None,
    backup_path: Union[str, None] = None,
    is_notebook: bool = False) -> OUTPUT_TYPES:

    """
    Heuristic/GPT evaluator.
    For more information, visit https://dbqr-qa.github.io/quickstart.html#evaluation

    Parameters `model`, `retry`, `openai_key`, `backup_path` are not used for
    the heuristic evaluator.

    The `is_notebook` parameter prevents tqdm from outputting 
    a new progress bar for every question.
    """

    def display(
        items: Iterable, 
        desc: Union[str, None] = None, 
        leave: bool = True) -> Iterable:

        if evaluator == 'heuristic' or (is_notebook and not leave):
            return items
    
        else:
            return tqdm(items, desc=desc, leave=leave)

    if evaluator not in EVALUATORS:
        raise KeyError(f'Invalid evaluators. The valid choices are {EVALUATORS}.')
    
    if evaluator.startswith('gpt'):
        if openai_key is None:
            raise RuntimeError("OpenAI API key is required for a GPT evaluator.")
        
        else:
            client = OpenAI(api_key=openai_key)

        scoring = evaluator.split('-')[1]
        prompt = GPT_PROMPT_BINARY if scoring == 'binary' else GPT_PROMPT_SCORE

        if backup_path is None:
            message = 'No backup path provided for the GPT evaluator.' \
                'The output will not be saved, and the evaluation ' \
                'cannot resume if terminated early.'
            
            logging.warning(message)

    scores, messages = {}, {}

    if backup_path is not None:
        Path(backup_path).mkdir(parents=True, exist_ok=True)
        backup_file = join(backup_path, f'gpt-evaluation-scores.json')

        if isfile(backup_file):
            try:
                with open(backup_file) as file:
                    backup = json.load(file)
                    scores = backup['scores']
                    messages = backup['messages']
            
            except json.decoder.JSONDecodeError:
                logging.warning('Unable to read the backup file.')

    for chat_id in display(questions.keys(), desc='Chat'):
        if chat_id not in scores or chat_id not in messages:
            scores[chat_id] = {}

            # Initialize GPT's input by explaining the task.
            messages[chat_id] = init_gpt_messages()

        # All conversations have ten questions.
        for question_id in display(range(1, 11), desc='Question', leave=False):
            question_key = str(question_id)

            if question_key in scores[chat_id]:
                continue

            question = questions[chat_id][question_key]
            answer = answers[chat_id][question_key]
            label = labels[chat_id][question_key]

            if evaluator == 'heuristic':
                scores[chat_id][question_key] = evaluate_heuristic(
                    answer, label)
            
            else:
                scores[chat_id][question_key], messages[chat_id] = evaulate_gpt(
                    question, answer, label, messages[chat_id], 
                    prompt, client, model, scoring, retry)

                if backup_path is not None:
                    with open(backup_file, 'w') as file:
                        json.dump({
                            'scores': scores,
                            'messages': messages
                        }, file, indent=2)

    all_scores = []

    for chat_scores in scores.values():
        all_scores += list(chat_scores.values())
    
    return float(sum(all_scores)) / len(all_scores), scores
