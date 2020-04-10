from typing import Optional, List
from pathlib import Path
from datetime import datetime, timedelta
from re import search
from numpy import ndarray
from skimage.io import imread
from skimage.color import rgb2gray
from .base import IdentifiedCollection, Unique, TimeStampElapsed
from .file_access import file_exists


class ImageFile(Unique, TimeStampElapsed):
    """
    An object to hold information about, and provide access to, a timestamped image file
    """
    def __init__(
        self,
        file_path: Path,
        timestamp: datetime = None,
        timestamp_initial: datetime = None,
        cache_image: bool = False
    ):
        super(ImageFile, self).__init__()
        self.file_path = file_path

        if timestamp is None:
            timestamp = self.timestamp_from_string(str(self.file_path.name))
        if timestamp_initial is None:
            timestamp_initial = timestamp

        self.timestamp = timestamp
        self.timestamp_initial = timestamp_initial
        self.cache_image = cache_image
        self.__image = None
        if self.cache_image:
            self.__image = ImageFile.__load_image(self.file_path)

    def __enter__(self):
        # Load and cache image ready for use
        if self.__image is None:
            self.__image = ImageFile.__load_image(self.file_path)

        return self

    def __exit__(self, exception_type, exception_value, traceback):
        # Remove cached images, unless required
        if not self.cache_image:
            self.__image = None

    @property
    def cache_image(self) -> bool:
        return self.__cache_image

    @cache_image.setter
    def cache_image(self, val: bool):
        self.__cache_image = val

    @property
    def image(self) -> ndarray:
        if self.cache_image and self.__image is not None:
            return self.__image.copy()
        else:
            return ImageFile.__load_image(self.file_path)

    @property
    def image_gray(self) -> ndarray:
        return rgb2gray(self.image)

    @property
    def file_path(self) -> Path:
        return self.__file_path

    @file_path.setter
    def file_path(self, val: Path):
        if not isinstance(val, Path):
            val = Path(val)

        if not file_exists(val):
            raise FileNotFoundError(f"The image file could not be found: {val}")

        self.__file_path = val

    @staticmethod
    def timestamp_from_exif(image_file: Path) -> Optional[datetime]:
        raise NotImplementedError()

    @staticmethod
    def timestamp_from_string(
        search_string: str,
        pattern: str =
        "(?P<year>\\d{4}).?(?P<month>[0-1][0-9]).?(?P<day>[0-3][0-9]).?(?P<hour>[0-2][0-9]).?(?P<minute>[0-5][0-9])"
    ) -> Optional[datetime]:
        """
        Attempts to read a datetime value from a string

        Requires a regex pattern with the following named pattern groups:
        year, month, day, hour, minute

        :param search_string: a string to check against the regex pattern
        :param pattern: a regex pattern used to match the datetime
        :returns: a datetime parsed from the string, if successful
        """
        if not len(search_string) > 0 or not len(pattern) > 0:
            raise ValueError("The search string or pattern must not be empty")

        result = search(pattern, search_string)
        if result:
            return datetime(
                year = int(result.groupdict()["year"]),
                month = int(result.groupdict()["month"]),
                day = int(result.groupdict()["day"]),
                hour = int(result.groupdict()["hour"]),
                minute = int(result.groupdict()["minute"])
            )
        else:
            return None

    @staticmethod
    def __load_image(file_path: Path, as_gray: bool = False, plugin: str = None, **plugin_args) -> ndarray:
        return imread(str(file_path), as_gray = as_gray, plugin = plugin, **plugin_args)


class ImageFileCollection(IdentifiedCollection):
    """
    Holds a collection of ImageFiles
    """
    @IdentifiedCollection.items.getter
    def items(self) -> List[ImageFile]:
        return sorted(self._IdentifiedCollection__items, key = lambda item: item.timestamp)

    @property
    def file_paths(self) -> List[datetime]:
        return [image_file.file_path for image_file in self.items]

    @property
    def timestamps(self) -> List[datetime]:
        return [image_file.timestamp for image_file in self.items]

    @property
    def timestamps_initial(self) -> List[datetime]:
        return [image_file.timestamp_initial for image_file in self.items]

    @timestamps_initial.setter
    def timestamps_initial(self, val: datetime):
        for image_file in self.items:
            image_file.timestamp_initial = val

    @property
    def timestamps_elapsed(self) -> List[timedelta]:
        return [image_file.timestamp_elapsed for image_file in self.items]

    @property
    def timestamps_elapsed_hours(self) -> List[float]:
        return [image_file.timestamp_elapsed_hours for image_file in self.items]

    @property
    def timestamps_elapsed_minutes(self) -> List[int]:
        return [image_file.timestamp_elapsed_minutes for image_file in self.items]

    @property
    def timestamps_elapsed_seconds(self) -> List[int]:
        return [image_file.timestamp_elapsed_seconds for image_file in self.items]

    def add(
        self,
        file_path: Path,
        timestamp: datetime = None,
        timestamp_initial: datetime = None,
        cache_image: bool = False
    ) -> ImageFile:
        """
        Create a new ImageFile and append it to the collection

        :param file_path: a Path object representing the image location
        :param timestamp: a datetime associated with the image
        :param timestamp_initial: a starting datetime used to calculate elapsed timestamps
        :param cache_image: load the image dynamically from file, or store in memory
        :returns: the new ImageFile instance
        """
        image_file = ImageFile(
            file_path = file_path,
            timestamp = timestamp,
            timestamp_initial = timestamp_initial,
            cache_image = cache_image
        )

        self.append(image_file)

        return image_file
