from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import pandas as pd


class DBInterface(ABC):
    """Database interface"""

    @abstractmethod
    def connect(self) -> None:
        """Connect to the database"""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the database"""
        pass

    @abstractmethod
    def add_record(self, record: Any) -> Any:
        """
        Insert a single ORM record into the database.

        :param record: the ORM model instance to insert
        :type record: Any
        :return: the inserted ORM model instance with refreshed fields
        :rtype: Any
        """
        pass
