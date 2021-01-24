from __future__ import annotations
from typing import Optional, List
from pathlib import Path
from datetime import datetime, timedelta
from re import search
from numpy import ndarray
from skimage.transform._geometric import GeometricTransform
from .base import IdentifiedCollection, Unique, TimeStampElapsed
from .file_access import file_exists


class ImageFile(Unique, TimeStampElapsed):
    """
    Holds information about, and provides access to, a timestamped image file
    """
    def __init__(
        self,
        file_path: Path,
        timestamp: datetime = None,
        timestamp_initial: datetime = None,
        cache_image: bool = False,
        align_image: bool = True
    ):
        super(ImageFile, self).__init__()

        self.file_path = file_path
        self.timestamp = timestamp or self.timestamp_from_string(str(self.file_path.name))
        self.timestamp_initial = timestamp_initial or self.timestamp
        self.cache_image = cache_image
        self.align_image = align_image
        self.alignment_transform = None
        self._image = None
        if self.cache_image:
            self._image = ImageFile._load_image(self.file_path)

    def __enter__(self):
        # Load and cache image ready for use
        if self._image is None:
            self._image = ImageFile._load_image(self.file_path)

        return self

    def __exit__(self, exception_type, exception_value, traceback):
        # Remove cached images, unless required
        if not self.cache_image:
            self._image = None

    @property
    def align_image(self) -> bool:
        return self._align_image

    @align_image.setter
    def align_image(self, val: bool):
        self._align_image = val

    @property
    def alignment_transform(self) -> Optional[GeometricTransform]:
        return self._alignment_transform

    @alignment_transform.setter
    def alignment_transform(self, val: GeometricTransform):
        self._alignment_transform = val

    @property
    def cache_image(self) -> bool:
        return self._cache_image

    @cache_image.setter
    def cache_image(self, val: bool):
        self._cache_image = val

    @property
    def image(self) -> ndarray:
        from imreg_dft import transform_img

        if self.cache_image and self._image is not None:
            image = self._image.copy()
        else:
            image = ImageFile._load_image(self.file_path)
        if self.align_image and self.alignment_transform is not None:
            scale = self.alignment_transform.scale if hasattr(self.alignment_transform, "scale") else 1
            image = transform_img(
                image,
                scale,
                self.alignment_transform.rotation,
                self.alignment_transform.translation,
                bgval = 0
            )

        return image

    @property
    def image_gray(self) -> ndarray:
        from skimage.color import rgb2gray

        return rgb2gray(self.image)

    @property
    def file_path(self) -> Path:
        return self._file_path

    @file_path.setter
    def file_path(self, val: Path):
        if not isinstance(val, Path):
            val = Path(val)

        if not file_exists(val):
            raise FileNotFoundError(f"The image file could not be found: {val}")

        self._file_path = val

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
    def _load_image(file_path: Path, as_gray: bool = False, plugin: str = None, **plugin_args) -> ndarray:
        from skimage.io import imread
        from .imaging import image_as_rgb

        while True:
            try:
                return image_as_rgb(imread(str(file_path), as_gray = as_gray, plugin = plugin, **plugin_args))
            except Exception:
                if not plugin:
                    # Retry imread once with a different plugin if none has been set
                    # PIL is quite tolerant and may be able to handle images that the default plugin cannot
                    plugin = "pil"
                else:
                    raise


class ImageFileCollection(IdentifiedCollection):
    """
    Holds a collection of ImageFiles
    """
    @IdentifiedCollection.items.getter
    def items(self) -> List[ImageFile]:
        return sorted(self._items.values(), key = lambda item: item.timestamp)

    @property
    def file_paths(self) -> List[datetime]:
        return [image_file.file_path for image_file in self.items if image_file.timestamp is not None]

    @property
    def timestamps(self) -> List[datetime]:
        return [image_file.timestamp for image_file in self.items if image_file.timestamp is not None]

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

    @classmethod
    def from_path(
        cls,
        path: Path,
        image_formats: List[str],
        timestamp_initial: datetime = None,
        cache_images: bool = False
    ) -> ImageFileCollection:
        """
        Build an ImageFileCollection from a directory containing image files

        :param path: the path to the image files on disk
        :param image_formats: a list of supported image format file extensions
        :param timestamp_initial: a timestamp used to calculate relative timestamps for ImageFiles
        :param cache_images: if images should be stored in memory as they are added to the collection
        :returns: a new ImageFileCollection populated with ImageFiles
        """
        from .file_access import get_files_by_type

        image_paths = get_files_by_type(path, image_formats)
        if not len(image_paths) > 0:
            raise FileNotFoundError(f"""No images could be found in the supplied folder path.
                Images are expected in these formats: {image_formats}""")

        # Store images as ImageFile objects
        # Timestamps are automatically read from filenames
        image_files = cls()
        for image_path in image_paths:
            image_files.add(
                file_path = image_path,
                timestamp = None,
                timestamp_initial = None,
                cache_image = False
            )

        # Check that timestamps were parsed correctly
        if image_files.count != len(image_files.timestamps):
            raise IOError("""Unable to load timestamps from all image filenames.
                Please check that images have a filename with YYYYMMDD_HHMM timestamps""")

        # Use first available timestamp if no initial timestamp is set
        image_files.timestamps_initial = timestamp_initial or image_files.timestamps[0]

        return image_files