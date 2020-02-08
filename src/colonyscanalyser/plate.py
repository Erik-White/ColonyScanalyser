from __future__ import annotations
from typing import Union, Dict, List, Tuple
from collections.abc import Collection
from pathlib import Path, PurePath
from numpy import ndarray
from .base import Identified, IdentifiedCollection, Named
from .geometry import Circle
from .file_access import save_to_csv


class Plate(Identified, IdentifiedCollection, Named, Circle):
    """
    An object to hold information about an agar plate and a collection of Colony objects
    """
    def __init__(
        self,
        id: int,
        diameter: float,
        edge_cut: float = 0,
        name: str = "",
        center: Union[Tuple[float, float], Tuple[float, float, float]] = None,
        colonies: list = None
    ):
        self.id = id
        self.diameter = diameter

        # Can't set argument default otherwise it is shared across all class instances
        if center is None:
            center = tuple()
        if colonies is None:
            colonies = list()

        # Set property defaults
        self.center = center
        self.items = colonies
        self.edge_cut = edge_cut
        self.name = name

    def __iter__(self):
        return iter([
            self.id,
            self.name,
            self.center,
            self.diameter,
            self.area,
            self.edge_cut,
            self.count
        ])

    @property
    def center(self) -> Union[Tuple[float, float], Tuple[float, float, float]]:
        return self.__center

    @center.setter
    def center(self, val: Union[Tuple[float, float], Tuple[float, float, float]]):
        self.__center = val

    @property
    def edge_cut(self) -> float:
        return self.__edge_cut

    @edge_cut.setter
    def edge_cut(self, val: float):
        self.__edge_cut = val

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
            self.items,
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
        for colony in self.items:
            for timepoint in colony.timepoints.values():
                colony_timepoints.append([colony.id, *timepoint])

        return self.__collection_to_csv(
            save_path,
            "_".join(filter(None, [f"plate{str(self.id)}", self.name.replace(" ", "_"), "colony", "timepoints"])),
            colony_timepoints,
            headers
        )

    def colonies_rename_sequential(self, start: int = 1) -> int:
        """
        Update the ID numbers of all colonies in the plate colony collection

        :param start: the new initial ID number
        :returns: the final ID number of the renamed sequence
        """
        for i, colony in enumerate(self.items, start = start):
            colony.id = i

        return i

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