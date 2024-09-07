from typing import Dict, Iterable, List, Tuple, Union

VALUE_TYPES = Union[str, int, float]
ANSWER_TYPES = Dict[str, Dict[str, Union[VALUE_TYPES, List[VALUE_TYPES], dict[str, VALUE_TYPES]]]]
