import pytest
import tempfile
from pathlib import Path

from colonyscanalyser.file_access import (file_exists,
                                get_files_by_type,
                                create_subdirectory,
                                move_to_subdirectory,
                                CompressionMethod,
                                file_compression,
                                load_file,
                                save_file,
                                save_to_csv
                                )

FILE_NON_EXISTANT = Path("")
SUB_DIR = Path("test_subdir")


def create_temp_file(dir, extension = None):
    return Path(tempfile.mkstemp(dir = dir, suffix = extension)[1])


class TestFileExists():
    def test_file_exists(self):
        assert file_exists(Path(__file__)) == True
    
    def test_file_does_not_exist(self):
        assert file_exists(FILE_NON_EXISTANT) == False

    def test_file_string(self):
        assert file_exists("") == False


class TestGetFilesByType():
    def test_return_list(self):
        # Check that a populated list is returned
        result = get_files_by_type(Path(__file__).parent)
        assert type(result) == list
        assert len(result) > 0

    def test_extension(self):
        # Check that files of a single extenion are returned
        suffix = ["py"]
        result = get_files_by_type(Path(__file__).parent, suffix)
        assert len(result) > 0
        assert result[0].suffix.replace('.', '') in suffix

    def test_extension_multiple(self, tmp_path):
        # Check that multiple types of extensions are returned
        suffixes = [".py", ".abc", ".xyz"]
        for suffix in suffixes:
            create_temp_file(tmp_path, suffix)
        result = get_files_by_type(tmp_path, suffixes)
        assert len(result) > 0
        result = [x.suffix for x in result]
        assert set(result) == set(suffixes)

    def test_file_string(self):
        result = get_files_by_type("")
        assert all(result) == True


class TestCreateSubdirectory():
    @pytest.fixture
    def result(self, tmp_path):
        yield create_subdirectory(tmp_path, SUB_DIR)

    def test_return_path(self, result):
        assert isinstance(result, Path)

    def test_return_existing(self, tmp_path, result):
        # Check that an existing subdir will not cause issues
        existing = create_subdirectory(tmp_path, SUB_DIR)
        assert result == existing

    def test_correct_parent(self, tmp_path, result):
        assert tmp_path == result.parent

    def test_readonly_exception(self, tmp_path):
        # Store the original folder permissions
        test_dir_chmod = Path.stat(tmp_path).st_mode

        # Create a new subdir and make it read + execute
        test_dir = create_subdirectory(tmp_path, SUB_DIR)
        test_dir.chmod(555)

        try:
            with pytest.raises(EnvironmentError):
                create_subdirectory(test_dir, SUB_DIR)
        finally:
            # Restore original folder permissions
            test_dir.chmod(test_dir_chmod)


class TestMoveToSubdirectory():
    def test_moved(self, tmp_path):
        temp_file = create_temp_file(tmp_path)
        result = move_to_subdirectory([temp_file], SUB_DIR)
        assert result[0].resolve() == tmp_path.joinpath(SUB_DIR, temp_file.name).resolve()

    def test_list_value_exception(self):
        with pytest.raises(ValueError):
            move_to_subdirectory(list(), SUB_DIR)

    def test_string_value_exception(self, tmp_path):
        with pytest.raises(ValueError):
            move_to_subdirectory([tmp_path], "")

    def test_write_exception(self, tmp_path):
        temp_file = create_temp_file(tmp_path)
        # Store the original folder permissions
        test_dir_chmod = Path.stat(tmp_path).st_mode

        # Create a new subdir and make it read + execute
        test_dir = create_subdirectory(tmp_path, SUB_DIR)
        test_dir.chmod(555)

        try:
            with pytest.raises(EnvironmentError):
                move_to_subdirectory([temp_file], test_dir)
        finally:
            # Restore original folder permissions
            test_dir.chmod(test_dir_chmod)


class TestFileCompression():
    @pytest.fixture(params=["r", "wb"])
    def file_access_modes(self, request):
        yield request.param

    def test_return_file_none(self):
        assert file_compression(FILE_NON_EXISTANT, "") == None

    def test_compression_readable(self, tmp_path, file_access_modes):
        for method in CompressionMethod:
            temp_file = create_temp_file(tmp_path, method.value)
            with file_compression(temp_file, method, file_access_modes) as outfile:
                if "w" in file_access_modes:
                    assert outfile.writable() is True
                else:
                    assert outfile.readable() is True


class TestLoadFile():
    @pytest.fixture
    def data(self):
        return [0, 1, 2, 3, 4]

    def test_return_none(self):
        assert load_file(FILE_NON_EXISTANT, CompressionMethod.NONE) == None

    def test_return_file(self, tmp_path, data):
        for method in CompressionMethod:
            with save_file(tmp_path.joinpath(method.name).with_suffix(method.value),
                            data,
                            method
                            ) as temp_file:
                assert load_file(temp_file, method) is not None


class TestSaveFile():
    @pytest.fixture
    def data(self):
        return [0, 1, 2, 3, 4]

    def test_save_compressed(self, tmp_path, data):
        for method in CompressionMethod:
            with save_file(tmp_path.joinpath(method.name).with_suffix(method.value),
                            data,
                            method
                            ) as temp_file:
                assert file_exists(temp_file) is True


class TestSaveToCSV():
    @pytest.fixture
    def headers(self):
        return ["one", "two", "three"]

    @pytest.fixture
    def data_list(self):
        return [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    @pytest.fixture
    def data_dict(self, data_list, headers):
        data_dict = dict.fromkeys(headers)
        for i, key in enumerate(data_dict.keys()):
            data_dict[key] = data_list[0][i]
        yield data_dict

    def test_list(self, tmp_path, headers, data_list):
        import csv

        result = save_to_csv(data_list, headers, tmp_path.joinpath("csv_list"))
        # Add headers to data
        data_list.insert(0, headers)
        
        # Check all rows were written correctly
        with open(result, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                assert [str(x) for x in data_list[i]] == row

    def test_dict(self, tmp_path, headers, data_dict):
        import csv
        
        result = save_to_csv(data_dict, headers, tmp_path.joinpath("csv_dict"))
        result_dict = dict.fromkeys(headers)
        
        with open(result, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                for key in result_dict.keys():
                    result_dict[key] = int(row[key])

        assert data_dict == result_dict

    def test_dict_view(self, tmp_path, headers, data_dict, data_list):
        import csv

        result = save_to_csv(data_dict.values(), headers, tmp_path.joinpath("csv_dict_view"))
        # Add headers to data
        data_list.insert(0, headers)
        
        # Check all rows were written correctly
        with open(result, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                assert [str(x) for x in data_list[i]] == row

    def test_iterable_unpack(self, tmp_path, headers, data_list):
        import csv
        
        # Create a generic object that will require unpacking
        class TestIterator:
            def __init__(self, prop):
                self.prop = prop
            
            def __iter__(self):
                return iter([
                    self.prop
                    ])

        data_iters = list()
        for row in data_list:
            data_iters.append(TestIterator(row[0]))

        result = save_to_csv(data_iters, headers, tmp_path.joinpath("csv_unpack"))
        data_iters.insert(0, TestIterator(headers))
        
        # Check all rows were written correctly
        with open(result, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i == 0:
                    data_iters[i].prop == row
                else:
                    assert [str(data_iters[i].prop)] == row

    def test_iterable(self, tmp_path):
        with pytest.raises(ValueError):
            save_to_csv(0, "", tmp_path.joinpath("test_csv"))

    def test_string_path(self, tmp_path):
        save_path = tmp_path.joinpath("test_csv")
        assert save_to_csv("", "", str(save_path)) == save_path.with_suffix(".csv")

    def test_ioerror(self, tmp_path):
        # Store the original folder permissions
        test_dir_chmod = Path.stat(tmp_path).st_mode

        # Create a new subdir and make it read + execute
        test_dir = create_subdirectory(tmp_path, SUB_DIR)
        test_dir.chmod(555)

        try:
            with pytest.raises(IOError):
                save_to_csv("", "", test_dir.joinpath("test_csv"))
        finally:
            # Restore original folder permissions
            test_dir.chmod(test_dir_chmod)