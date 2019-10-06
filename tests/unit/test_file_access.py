import pytest
import tempfile
from pathlib import Path

from scanlag.file_access import (file_exists,
                                get_files_by_type,
                                create_subdirectory,
                                move_to_subdirectory,
                                CompressionMethod,
                                file_compression,
                                load_file,
                                save_file
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
