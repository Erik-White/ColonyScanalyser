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
