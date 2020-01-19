class Identified:
    """
    An object with a integer ID number
    """
    def __init__(self, id: int):
        self.id = id

    @property
    def id(self):
        return self.__id

    @id.setter
    def id(self, val: int):
        if self.__id_is_valid(self, val):
            self.__id = val
        else:
            raise ValueError(f"'{val}' is not a valid id. An id must be a non-negative integer'")

    @staticmethod
    def __id_exists(self, collection: list, id: int):
        return any(id == existing.id for existing in collection)

    @staticmethod
    def __id_is_valid(self, val: int):
        return isinstance(val, int) and val >= 0


class Named:
    """
    An object with a string identifier
    """
    def __init__(self, name: str):
        self.name = name

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, val: str):
        self.__name = val


class Unique(Identified):
    """
    An object with a incremented integer ID number
    """
    id_count = 0

    def __init__(self):
        self.__id = self.id_increment()

    @property
    def id(self):
        return self.__id

    @id.setter
    def id(self, val: int):
        """
        Overrides base method to prevent setting id
        """
        pass

    @classmethod
    def id_increment(self):
        self.id_count += 1
        return self.id_count