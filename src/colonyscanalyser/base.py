from collections.abc import Collection


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
        if self.__id_is_valid(self, val):
            self.__id = val
        else:
            raise ValueError(f"'{val}' is not a valid id. An id must be a non-negative integer'")

    @staticmethod
    def __id_exists(self, collection: Collection, id: int) -> bool:
        """
        Verifies if an object in a collection matches the specified ID number

        :param collection: a collection of objects (List, Dict etc)
        :param id: an ID number to locate
        :returns: True if an object with id exists in the collection
        """
        return any(id == existing.id for existing in collection)

    @staticmethod
    def __id_is_valid(self, id: int) -> bool:
        """
        Verifies if a value conforms to the requirements for an ID number

        An ID number is an integer with a value greater than zero

        :param id: an ID number to verify
        :returns: True if the value conforms to the requirements for an ID number
        """
        return isinstance(id, int) and id > 0


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
        Overrides base method to prevent setting id
        """
        pass

    @classmethod
    def id_increment(self) -> int:
        """
        Increments the built-in ID counter

        :returns: the auto incremented ID number
        """
        self.id_count += 1

        return self.id_count