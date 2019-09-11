def get_images(path, filetypes = "tif"):
    return sorted(path.glob("*." + filetypes))

def create_folderpath(filepath):
    """
    Create any folder that is needed in a filepath
    """
    import os
    os.makedirs(filepath, exist_ok=True)


def separate_filepath(filepath, return_folderpath = False):
    """
    Separates the folderpath and filename from a filepath

    :returns: either a filename (default) or folderpath
    """
    import os
    (folderpath, filename) = os.path.split(os.path.abspath(filepath))
    if return_folderpath:
        return folderpath
    else:
        return filename


def get_subfoldername(data_folder, row, col):
    import os
    return data_folder+'_'.join(['row', str(row), 'col', str(col)])+os.path.sep


def file_exists(filepath):
    """
    Checks whether a file exists and contains data
    """
    import os
    if os.path.isfile(filepath) and os.path.getsize(filepath) > 0:
        return True


def mkdir_plates(data_folder, lattice):
    import os
    '''Make subfolders for single plates'''
    for row in range(lattice[0]):
        for col in range(lattice[1]):
            subfn = get_subfoldername(data_folder, row + 1, col + 1)
            if not os.path.isdir(subfn):
                os.mkdir(subfn)

def move_to_subdirectory(file_list, subfolder_name):
    """
    Move all files in a list to a subdirectory

    The subdirectory is relative to their current location.
    If the subdirectory will be created if it does not already exist

    :param file_list: a list of path objects
    :param subfolder_name: a string representing a new or existing folder
    :returns: an updated list of path objects
    """
    from pathlib import Path

    if not len(file_list) > 0:
        raise ValueError("The supplied list of path objects is empty")
    if not len(subfolder_name) > 0:
        raise ValueError("A sub folder name must be supplied")

    files = []
    parent_path = file_list[0].resolve().parent

    # Create a subdirectory relative to the parent path, if needed
    parent_path.joinpath(subfolder_name).mkdir(exist_ok = True)
    if not parent_path.joinpath(subfolder_name).exists():
        raise EnvironmentError("Unable to create new subfolder:", subfolder_name)

    # Move files to subdirectory and build a new list of files
    for file in file_list:
        new_path = parent_path.joinpath(subfolder_name, file.name)
        file.replace(new_path)
        files.append(new_path)

    return files