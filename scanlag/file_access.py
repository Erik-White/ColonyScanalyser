def file_exists(filepath):
    """
    Check whether a file exists and contains data
    """
    from pathlib import Path
    
    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    if Path.exists(filepath) and Path.is_file(filepath) and Path.stat(filepath).st_size > 0:
        return True
    
    return False


def get_files_by_type(path, file_extensions = ["*"]):
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
        path_list.extend(path.glob("*." + file_extension))

    return sorted(path_list)


def create_subdirectory(parent_path, subdirectory):
    """
    Create a subdirectory relative to the parent, if required

    :param parent_path: a path object
    :param subdirectory: a string or path object representing a new or existing folder
    :returns: the new directory path object
    """
    from pathlib import Path

    subdir_path = parent_path.joinpath(subdirectory)
    try:
        subdir_path.mkdir(exist_ok = True)
    except:
        # There are many reasons creating a directory may fail
        # Invalid filename, permissions, etc
        raise EnvironmentError("Unable to create new subfolder:", subdirectory)

    return subdir_path


def move_to_subdirectory(file_list, subdirectory):
    """
    Move all files in a list to a subdirectory

    The subdirectory is relative to their current location
    The subdirectory will be created if it does not already exist

    :param file_list: a list of path objects
    :param subdirectory: a string or path object representing a new or existing folder
    :returns: an updated list of path objects
    """
    from pathlib import Path

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
    except:
        raise EnvironmentError("Unable to move files to subdirectory:", sub_dir)

    return files


from enum import Enum, auto
class CompressionMethod(Enum):
    BZ2 = ".pbz2"
    GZIP = ".gz"
    LZMA = ".xz"
    PICKLE = ".pyc"
    NONE = ""

    def __str__(self):
        return str(self.value)


def file_compression(file_path, compression, access_mode = "r"):
    """
    Allows access to a file using the desired compression method
    """
    from pathlib import Path

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


def load_file(file_path, compression, access_mode = "rb", pickle = True):
    """
    Load compressed data from a file
    """
    from numpy import load
    
    try:
        file_open = file_compression(file_path, compression, access_mode)
    except:
        return None
    
    return load(file_open, allow_pickle = pickle)


def save_file(file_path, data, compression):
    """
    Save data to specified file

    Returns the saved file path if successful
    """
    import pickle

    completed = None
    
    try:
        with file_compression(file_path, compression, "wb") as outfile:
            pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)
            completed = file_path
    finally:
        return completed