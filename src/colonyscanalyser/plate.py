from collections.abc import Collection
from .base import Identified, Named
from .geometry import Circle
from .colony import Colony


class Plate(Identified, Named, Circle):
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
        self.description = ""
        self.edge_cut = 0
        self.name = ""

    def __iter__(self):
        return iter([
            self.id,
            self.name,
            self.center,
            self.diameter,
            self.area,
            self.edge_cut,
            self.colony_count
        ])

    @property
    def colonies(self):
        return self.__colonies

    @colonies.setter
    def colonies(self, val: Collection):
        if isinstance(val, list):
            self.__colonies = val
        elif isinstance(val, Collection) and not isinstance(val, str):
            self.__colonies = [colony for colony in val]
        else:
            raise ValueError("Colonies must be supplied as a List or other Collection")

    @property
    def colony_count(self):
        return len(self.colonies)

    @property
    def edge_cut(self):
        return self.__edge_cut

    @edge_cut.setter
    def edge_cut(self, val: float):
        self.__edge_cut = val

    def add_colony(self, colony: Colony):
        """
        Append a Colony object to the plate colony collection

        :param colony: a Colony object
        """
        if not self.colony_exists(colony):
            self.colonies.append(colony)
        else:
            raise ValueError(f"A colony with ID #{colony.id} already exists")

    def colony_exists(self, colony: Colony):
        """
        Check if a colony exists in the plate colony collection

        :param colony: a Colony object
        :returns: True if a colony is found with matching ID
        """
        return self.colony_id_exists(colony.id)

    def colony_id_exists(self, colony_id: int):
        """
        Check if a colony with the specified ID number exists in the plate colony collection

        :param colony_id: a Colony object id number
        :returns: True if a colony is found with matching ID
        """
        return self._Identified__id_exists(self, self.colonies, colony_id)

    def colonies_rename_sequential(self, start: int = 1):
        """
        Update the ID numbers of all colonies in the plate colony collection

        :param start: the new initial ID number
        :returns: the final ID number of the renamed sequence
        """
        for i, colony in enumerate(self.colonies, start = start):
            colony.id = i

        return i

    def get_colony(self, colony_id: int):
        """
        Returns a colony with the specified ID number from the plate colony collection

        :param colony_id: a Colony ID number
        :returns: a Colony object, if found
        """
        for colony in self.colonies:
            if colony.id == colony_id:
                return colony

        return None

    def remove_colony(self, colony_id: int):
        """
        Remove a Colony object from the plate colony collection

        :param colony_id: a Colony object ID
        """
        if self.colony_id_exists(colony_id):
            for i, colony in enumerate(self.colonies):
                if colony.id == colony_id:
                    self.colonies.remove(colony)
        else:
            raise KeyError(f"No colony with ID #{colony_id} could be found")