def file_exists(filepath):
    """
    Check whether a file exists and contains data
    """
    from pathlib import Path
    
    if Path.exists(filepath) and Path.stat(filepath).st_size > 0:
        return True


def get_files_by_type(path, file_extensions = "*"):
    """
    Get a list of path objects of a given filetype(s)

    :param path: a path object directory to search
    :param filetypes: a comma separated list of file extensions
    :returns: a list of matching path objects, sorted by filename
    """
    from pathlib import Path
    
    path_list = []
    file_extensions = file_extensions.split(",")

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
    subdir_path.mkdir(exist_ok = True)
    if not subdir_path.exists():
        raise EnvironmentError("Unable to create new subfolder:", subdirectory)
    return subdir_path


def move_to_subdirectory(file_list, subdirectory):
    """
    Move all files in a list to a subdirectory

    The subdirectory is relative to their current location.
    If the subdirectory will be created if it does not already exist

    :param file_list: a list of path objects
    :param subdirectory: a string or path object representing a new or existing folder
    :returns: an updated list of path objects
    """
    from pathlib import Path

    if not len(file_list) > 0:
        raise ValueError("The supplied list of path objects is empty")
    if not len(subdirectory) > 0:
        raise ValueError("A sub folder name must be supplied")

    files = []
    parent_path = file_list[0].resolve().parent

    # Create a subdirectory relative to the parent path, if needed
    create_subdirectory(parent_path, subdirectory)

    # Move files to subdirectory and build a new list of files
    for file in file_list:
        new_path = parent_path.joinpath(subdirectory, file.name)
        file.replace(new_path)
        files.append(new_path)

    return files