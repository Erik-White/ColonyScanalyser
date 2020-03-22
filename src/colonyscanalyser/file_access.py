from typing import Optional, Union, List
from enum import Enum
from collections.abc import Collection
from pathlib import Path


def file_exists(file_path: Path) -> bool:
    """
    Check whether a file exists and contains data

    :param file_path: a Path object representing a file
    :returns: True if the file exists and contains data
    """
    from pathlib import Path

    if not isinstance(file_path, Path):
        file_path = Path(file_path)

    if Path.exists(file_path) and Path.is_file(file_path) and Path.stat(file_path).st_size > 0:
        return True

    return False


def file_safe_name(file_name: List[str], separator: str = "_") -> str:
    """
    Converts a list of string to a safe file name

    :param file_name: a list of strings that make up the complete filename
    :param separator: a character to place in between the items of file_name
    :returns: a new filename string
    """
    safe_names = [val.replace(" ", separator) for val in file_name]

    return separator.join(filter(None, safe_names))


def get_files_by_type(path: Path, file_extensions: List[str] = ["*"]) -> List[Path]:
    """
    Get a list of path objects of a given filetype(s)

    :param path: a path object directory to search
    :param filetypes: a list of file extension strings
    :returns: a list of matching path objects, sorted by filename
    """
    from pathlib import Path

    if not isinstance(path, Path):
        path = Path(path)

    # Remove separator if passed in argument
    file_extensions = [x.replace(".", "") for x in file_extensions]

    path_list = []
    for file_extension in file_extensions:
        path_list.extend([x for x in path.glob("*." + file_extension) if not str(x.name).startswith(".")])

    return sorted(path_list)


def create_subdirectory(parent_path: Path, subdirectory: Union[Path, str]) -> Path:
    """
    Create a subdirectory relative to the parent, if required

    :param parent_path: a path object
    :param subdirectory: a string or path object representing a new or existing folder
    :returns: the new directory path object
    """
    subdir_path = parent_path.joinpath(subdirectory)

    try:
        subdir_path.mkdir(exist_ok = True)
    except Exception:
        # There are many reasons creating a directory may fail
        # Invalid filename, permissions, etc
        raise EnvironmentError(f"Unable to create new subfolder: {subdirectory}")

    return subdir_path


def move_to_subdirectory(file_list: List[Path], subdirectory: Union[Path, str]) -> List[Path]:
    """
    Move all files in a list to a subdirectory

    The subdirectory is relative to their current location
    The subdirectory will be created if it does not already exist

    :param file_list: a list of path objects
    :param subdirectory: a string or path object representing a new or existing folder
    :returns: an updated list of path objects
    """
    if not len(file_list) > 0:
        raise ValueError("The supplied list of path objects is empty")
    if not len(str(subdirectory)) > 0:
        raise ValueError("A sub folder name must be supplied")

    files = list()
    parent_path = file_list[0].resolve().parent

    # Create a subdirectory relative to the parent path, if needed
    sub_dir = create_subdirectory(parent_path, subdirectory)

    try:
        # Move files to subdirectory and build a new list of files
        for file in file_list:
            file.replace(sub_dir.joinpath(file.name))
            files.append(sub_dir.joinpath(file.name))
    except Exception:
        raise EnvironmentError(f"Unable to move files to subdirectory: {sub_dir}")

    return files


class CompressionMethod(Enum):
    """
    An Enum representing different types of data compression

    The enum values are the corresponding file suffixes
    """
    BZ2 = ".pbz2"
    GZIP = ".gz"
    LZMA = ".xz"
    PICKLE = ".pyc"
    NONE = ""


def file_compression(file_path: Path, compression: CompressionMethod, access_mode: str = "r") -> Optional[object]:
    """
    Allows access to a file using the desired compression method

    :param file_path: a Path object
    :param compression: a CompressionMethod enum
    :param access_mode: the access mode for opening the file
    :returns: a file object or datastream, if successful, or None
    """
    if compression == CompressionMethod.BZ2:
        import bz2
        return bz2.BZ2File(file_path.with_suffix(CompressionMethod.BZ2.value), mode = access_mode)
    elif compression == CompressionMethod.GZIP:
        import gzip
        return gzip.GzipFile(file_path.with_suffix(CompressionMethod.GZIP.value), mode = access_mode)
    elif compression == CompressionMethod.LZMA:
        import lzma
        return lzma.LZMAFile(file_path.with_suffix(CompressionMethod.LZMA.value), mode = access_mode)
    elif compression == CompressionMethod.PICKLE:
        return open(file_path.with_suffix(CompressionMethod.PICKLE.value), mode = access_mode)
    elif compression == CompressionMethod.NONE:
        return open(file_path, mode = access_mode)

    return None


def load_file(
    file_path: Path,
    compression: CompressionMethod,
    access_mode: str = "rb",
    pickle: bool = True
) -> Optional[object]:
    """
    Load compressed data from a file

    :param file_path: a Path object
    :param compression: a CompressionMethod enum
    :param access_mode: the access mode for opening the file
    :param pickle: optionally disallow loading object arrays for security reasons
    :returns: the loaded data object, if successful, or None
    """
    from numpy import load

    try:
        file_open = file_compression(file_path, compression, access_mode)
    except Exception:
        return None

    return load(file_open, allow_pickle = pickle)


def save_file(file_path: Path, data: Collection, compression: CompressionMethod) -> Path:
    """
    Save data to specified file

    Returns the saved file path if successful

    :param file_path: a Path object
    :param data: the output data to save
    :param compression: a CompressionMethod enum
    :returns: a Path object representing the saved file, if successful
    """
    import pickle

    completed = None

    try:
        with file_compression(file_path, compression, "wb") as outfile:
            pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)
            completed = file_path
    finally:
        return completed


def save_to_csv(data: Collection, headers: List[str], save_path: Union[Path, str], delimiter: str = ",") -> Path:
    """
    Save data to CSV files on disk

    Accepts lists, dictionaries and generic iterable objects

    :param data: an iterable object
    :param headers: column headers for the CSV file
    :param save_path: the save location as a path object or string
    :returns: a path object representing the new file, if successful
    """
    import csv
    from pathlib import Path
    from collections.abc import Collection, Iterable, MappingView

    if not isinstance(data, Iterable):
        raise ValueError("The data object must be iterable e.g. a list")

    if isinstance(save_path, str):
        save_path = Path(save_path)
    save_path = save_path.with_suffix(".csv")

    try:
        with open(save_path, 'w') as outfile:
            if isinstance(data, dict):
                # Dictionary values are assigned by key to column headers
                writer = csv.DictWriter(
                    outfile,
                    delimiter = delimiter,
                    fieldnames = headers
                )
                writer.writeheader()
                data = [data]
            else:
                writer = csv.writer(outfile, delimiter = delimiter)
                writer.writerow(headers)

            # Check if iterable contains objects that need unpacking
            unpack = False
            for row in data:
                if isinstance(row, Iterable) and not isinstance(row, Collection):
                    unpack = True
                break

            # View objects are not iterable and need wrapping
            if isinstance(data, MappingView) or unpack:
                data = [data]

            # Write the data
            if unpack:
                writer.writerows(*data)
            else:
                writer.writerows(data)

    except (Exception, csv.Error):
        raise IOError(f"Unable to save data to CSV file at {save_path}")

    return save_path