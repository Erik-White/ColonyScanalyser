from collections.abc import Iterable
from .geometry import Circle


class Plate(Circle):
    """
    An object to hold information about an agar plate and a collection of Colony objects
    """
    def __init__(self, id: int, diameter: float, colonies: list = None):
        self.id = id
        self.diameter = diameter

        # Can't set argument default otherwise it is shared across all class instances
        if colonies is None:
            colonies = list()

        # Set property defaults
        self.colonies = colonies
        self.center = None
        self.description = ""
        self.edge_cut = 0
        self.name = ""

    def __iter__(self):
        return iter([
            self.id,
            self.name,
            self.description,
            self.center,
            self.diameter,
            self.radius,
            self.circumference,
            self.area,
            self.edge_cut,
            self.colony_count
        ])

    @property
    def center(self):
        return self.__center

    @center.setter
    def center(self, val: tuple):
        self.__center = val

    @property
    def colonies(self):
        return self.__colonies

    @colonies.setter
    def colonies(self, val: Iterable):
        if isinstance(val, list):
            self.__colonies = val
        elif isinstance(val, Iterable) and not isinstance(val, str):
            self.__colonies = [colony for colony in val]
        else:
            raise ValueError("Colonies must be supplied as a List or other iterable")

    @property
    def colony_count(self):
        return len(self.colonies)

    @property
    def description(self):
        return self.__description

    @description.setter
    def description(self, val: str):
        self.__description = val

    @property
    def edge_cut(self):
        return self.__edge_cut

    @edge_cut.setter
    def edge_cut(self, val: float):
        self.__edge_cut = val

    @property
    def id(self):
        return self.__id

    @id.setter
    def id(self, val: float):
        if isinstance(val, int) and val >= 0:
            self.__id = val
        else:
            raise ValueError(f"'{val}' is not a valid id. An id must be a non-negative integer'")

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, val: str):
        self.__name = val

    def append_colony(self, colony: object):
        """
        Add a Colony object to the plate colony collection

        :param colony: a Colony object
        """
        if not self.colony_exists(colony):
            self.colonies.append(colony)
        else:
            raise ValueError(f"A colony with ID #{colony.id} already exists")

    def colony_exists(self, colony: object):
        """
        Check if a colony exists in the plate colony collection

        :param colony: a Colony object
        :returns: True if a colony is found with matching ID
        """
        return self.__id_exists(self, self.colonies, colony.id)

    def colonies_rename_sequential(self, start: int = 1):
        """
        Update the ID numbers of all colonies in the plate colony collection

        :param start: the new initial ID number
        """
        for i, colony in enumerate(self.colonies, start = start):
            colony.id = i

    @staticmethod
    def __id_exists(self, collection: list, id: int):
        return any(id == existing.id for existing in collection)
