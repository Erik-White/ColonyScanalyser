from typing import List
from collections.abc import Collection
from pathlib import Path, PurePath
from .base import Identified, Named
from .geometry import Circle
from .colony import Colony
from .file_access import save_to_csv


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

    def colonies_to_csv(self, save_path: Path, headers: List[str] = None) -> Path:
        """
        Output the data from the colonies collection to a CSV file

        :param save_path: the location to save the CSV data file
        :param headers: a list of strings to use as column headers
        :returns: a Path representing the new file, if successful
        """
        if headers is None:
            headers = [
                "Colony ID",
                "Time of appearance",
                "Time of appearance (elapsed minutes)",
                "Center point averaged (row, column)",
                "Colour averaged name",
                "Colour averaged (R,G,B)",
                "Growth rate average",
                "Growth rate",
                "Doubling time average (minutes)",
                "Doubling times (minutes)",
                "First detection (elapsed minutes)",
                "First center point (row, column)",
                "First area (pixels)",
                "First diameter (pixels)",
                "Final detection (elapsed minutes)",
                "Final center point (row, column)",
                "Final area (pixels)",
                "Final diameter (pixels)"
            ]

        return self.__collection_to_csv(
            save_path,
            "_".join(filter(None, [f"plate{str(self.id)}", self.name.replace(" ", "_"), "colonies"])),
            self.colonies,
            headers
        )

    def colonies_timepoints_to_csv(self, save_path: Path, headers: List[str] = None) -> Path:
        """
        Output the data from the timepoints in the colonies collection to a CSV file

        :param save_path: the location to save the CSV data file
        :param headers: a list of strings to use as column headers
        :returns: a Path representing the new file, if successful
        """
        if headers is None:
            headers = [
                "Colony ID",
                "Date/Time",
                "Elapsed time (minutes)",
                "Area (pixels)",
                "Center (row, column)",
                "Diameter (pixels)",
                "Perimeter (pixels)",
                "Color average (R,G,B)"
            ]

        # Unpack timepoint properties to a flat list
        colony_timepoints = list()
        for colony in self.colonies:
            for timepoint in colony.timepoints.values():
                colony_timepoints.append([colony.id, *timepoint])

        return self.__collection_to_csv(
            save_path,
            "_".join(filter(None, [f"plate{str(self.id)}", self.name.replace(" ", "_"), "colony", "timepoints"])),
            colony_timepoints,
            headers
        )

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
        return self._Identified__id_exists(self.colonies, colony_id)

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

    @staticmethod
    def __collection_to_csv(save_path: Path, file_name: str, data: Collection, headers: List[str] = None) -> Path:
        """
        Output the data from the timepoints in the colonies collection to a CSV file

        :param save_path: the location to save the CSV data file
        :param file_name: the name of the CSV data file
        :param data: a collection of iterables to output as rows to the CSV file
        :param headers: a list of strings to use as column headers
        :returns: a Path representing the new file, if successful
        """
        # Check that a path has been specified and can be found
        if not isinstance(save_path, Path):
            save_path = Path(save_path)
        if not save_path.exists() or str(PurePath(save_path)) == ".":
            raise FileNotFoundError(f"The path '{str(save_path)}' could not be found. Please specify a different save path")

        return save_to_csv(
            data,
            headers,
            save_path.joinpath(file_name)
        )