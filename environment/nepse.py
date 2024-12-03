from typing import Any

from environment.base import Environment


class Nepse(Environment):
    def __init__(self, /, **data: Any):
        super().__init__(**data)
