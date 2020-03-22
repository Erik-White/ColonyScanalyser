from typing import Type, TypeVar, Optional, List
from collections.abc import Collection
from datetime import datetime, timedelta


class Identified:
    """
    An object with a integer ID number
    """
    def __init__(self, id: int):
        self.id = id

    @property
    def id(self) -> int:
        return self.__id

    @id.setter
    def id(self, val: int):
        if self.__id_is_valid(val):
            self.__id = val
        else:
            raise ValueError(f"'{val}' is not a valid id. An id must be a non-negative integer'")

    @staticmethod
    def __id_exists(collection: Collection, id: int) -> bool:
        """
        Verifies if an object in a collection matches the specified ID number

        :param collection: a collection of objects (List, Dict etc)
        :param id: an ID number to locate
        :returns: True if an object with id exists in the collection
        """
        return any(id == existing.id for existing in collection)

    @staticmethod
    def __id_is_valid(id: int) -> bool:
        """
        Verifies if a value conforms to the requirements for an ID number

        An ID number is an integer with a value greater than zero

        :param id: an ID number to verify
        :returns: True if the value conforms to the requirements for an ID number
        """
        return isinstance(id, int) and id > 0


class IdentifiedCollection:
    """
    An collection of Identified objects with generic methods for modifying the collection
    """
    T = TypeVar("T", bound = Identified)

    def __init__(self, items: Collection = None):
        self.items = items

    @property
    def count(self) -> int:
        return len(self.items)

    @property
    def items(self) -> List["T"]:
        """
        Returns a sorted list of items from the collection

        A copy is returned, preventing direct changes to the collection
        """
        return sorted(self.__items, key = lambda item: item.id)

    @items.setter
    def items(self, val: Collection):
        if isinstance(val, dict):
            val = list(val.values())
        if val is None:
            self.__items = list()
        elif isinstance(val, Collection) and not isinstance(val, str):
            self.__items = val.copy()
        else:
            raise ValueError(f"Items must be supplied as a valid Collection, not {type(val)}")

    def add(self, id: int) -> "T":
        """
        Create a new instance of T and append it to the collection

        :param id: a valid Identified ID number
        :returns: a new instance of T
        """
        item = Identified(id = id)

        self.append(item)

        return item

    def append(self, item: Type[T]):
        """
        Append an item to the collection

        :param item: the object to append to the collection
        """
        if not self.exists(item):
            self.__items.append(item)
        else:
            raise ValueError(f"An item with ID #{item.id} already exists")

    def exists(self, item: Type[T]) -> bool:
        """
        Check if an item exists in the item collection

        :param item: an instance of T
        :returns: True if an item is found with matching ID
        """
        return self.id_exists(item.id)

    def id_exists(self, id: int) -> bool:
        """
        Check if an item with the specified ID number exists in the item collection

        :param id: a valid Identified id number
        :returns: True if an item is found with matching ID
        """
        return Identified._Identified__id_exists(self.items, id)

    def get_item(self, id: int) -> Optional["T"]:
        """
        Returns an item with the specified ID number from the item collection

        :param id: a valid Identified ID number
        :returns: an item from the collection, if found
        """
        for item in self.items:
            if item.id == id:
                return item

        return None

    def remove(self, id: int):
        """
        Remove an item from the collection

        :param id: a valid Identified ID number
        """
        if self.id_exists(id):
            for item in self.items:
                if item.id == id:
                    self.__items.remove(item)
        else:
            raise KeyError(f"No item with ID #{id} could be found")


class Named:
    """
    An object with a string identifier
    """
    def __init__(self, name: str):
        self.name = name

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, val: str):
        self.__name = str(val)


class Unique(Identified):
    """
    An object with a auto incremented integer ID number
    """
    id_count = 0

    def __init__(self):
        self._Identified__id = self.id_increment()

    @Identified.id.setter
    def id(self, val: int):
        """
        Overrides base method to make id read-only
        """
        pass

    def id_increment(self) -> int:
        """
        Increments the built-in ID counter

        :returns: the auto incremented ID number
        """
        Unique.id_count += 1

        return Unique.id_count


class TimeStamped:
    def __init__(self, timestamp: datetime = None):
        if timestamp is None:
            timestamp = datetime.now()

        self.timestamp = timestamp

    @property
    def timestamp(self) -> datetime:
        return self.__timestamp

    @timestamp.setter
    def timestamp(self, val: datetime):
        self.__timestamp = val


class TimeStampElapsed(TimeStamped):
    def __init__(self, timestamp: datetime = None, timestamp_initial: datetime = None):
        if timestamp is None:
            timestamp = datetime.now()
        if timestamp_initial is None:
            timestamp_initial = timestamp

        self._TimeStamped__timestamp = timestamp
        self.timestamp_initial = timestamp_initial

    @property
    def timestamp_elapsed(self) -> timedelta:
        return self.timestamp - self.timestamp_initial

    @property
    def timestamp_elapsed_hours(self) -> float:
        return (self.timestamp_elapsed_seconds / 60) / 60

    @property
    def timestamp_elapsed_minutes(self) -> int:
        return int(self.timestamp_elapsed_seconds / 60)

    @property
    def timestamp_elapsed_seconds(self) -> int:
        return self.timestamp_elapsed.total_seconds()

    @property
    def timestamp_initial(self) -> datetime:
        return self.__timestamp_initial

    @timestamp_initial.setter
    def timestamp_initial(self, val: datetime):
        self.__timestamp_initial = val