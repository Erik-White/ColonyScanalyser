import unittest
import tempfile
import shutil
from pathlib import Path

import scanlag.file_access as file_access

class TestFileExists(unittest.TestCase):
    def test_return_bool(self):
        result = file_access.file_exists(Path(""))
        self.assertIs(type(result), bool)
    
    def test_return_true(self):
        self.assertTrue(file_access.file_exists(Path(__file__)))
    
    def test_return_false(self):
        self.assertFalse(file_access.file_exists(Path("")))

class TestGetFilesByType(unittest.TestCase):
    def test_return_list(self):
        result = file_access.get_files_by_type(Path(__file__))
        self.assertIs(type(result), list)
    
    def test_return_any(self):
        result = file_access.get_files_by_type(Path(__file__).parent)
        self.assertTrue(result)
    
    def test_return_single(self):
        result = file_access.get_files_by_type(Path(__file__).parent, "py")
        self.assertTrue(result)

class TestCreateSubdirectory(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_return_path(self):
        result = file_access.create_subdirectory(self.test_dir, Path(tempfile.gettempprefix()))
        self.assertIsInstance(result, Path)

    def test_return_existing(self):
        result = file_access.create_subdirectory(self.test_dir, Path(tempfile.gettempprefix()))
        result_existing = file_access.create_subdirectory(self.test_dir, Path(tempfile.gettempprefix()))
        self.assertEqual(result, result_existing)

    def test_correct_parent(self):
        result = file_access.create_subdirectory(self.test_dir, Path(tempfile.gettempprefix()))
        self.assertEqual(result.parent, self.test_dir)

    def test_readonly_exception(self):
        # Store the original folder permissions
        test_dir_chmod = Path.stat(self.test_dir).st_mode
        # Make it read + execute
        self.test_dir.chmod(555)
        try:
            with self.assertRaises(EnvironmentError):
                file_access.create_subdirectory(self.test_dir, Path(tempfile.gettempprefix()))
        finally:
            # Restore original folder permissions
            self.test_dir.chmod(test_dir_chmod)

class TestMoveToSubdirectory(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory and files
        self.test_dir = Path(tempfile.mkdtemp())
        self.sub_dir = "test_subdir"
        self.test_file_list = self.create_temp_files(self.test_dir)

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def create_temp_files(self, parent_dir):
        file_list = list()
        for i in range(5):
            temp_file = Path(tempfile.mkstemp(dir = parent_dir)[1])
            file_list.append(temp_file)
        return file_list

    def test_return_list(self):
        result = file_access.move_to_subdirectory(self.test_file_list, self.sub_dir)
        self.assertIs(type(result), list)

    def test_moved_parent(self):
        result = file_access.move_to_subdirectory(self.test_file_list, self.sub_dir)
        for file in result:
            self.assertEqual(file.parent.resolve(), self.test_dir.joinpath(self.sub_dir).resolve())

    def test_list_value_exception(self):
        with self.assertRaises(ValueError):
            file_access.move_to_subdirectory(list(), "")

    def test_string_value_exception(self):
        with self.assertRaises(ValueError):
            file_access.move_to_subdirectory(self.test_file_list, "")

class TestFileCompression(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory and files
        self.test_dir = Path(tempfile.mkdtemp())
        self.temp_file = Path(tempfile.mkstemp(suffix = ".pyc", dir = self.test_dir)[1])

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_return_none(self):
        self.assertIsNone(file_access.file_compression(self.temp_file, ""))

    def test_access_readable(self):
        with file_access.file_compression(self.temp_file, file_access.CompressionMethod.NONE, "r") as outfile:
            self.assertTrue(outfile.readable)

    def test_access_writeable(self):
        with file_access.file_compression(self.temp_file, file_access.CompressionMethod.NONE, "wb") as outfile:
            self.assertTrue(outfile.writable)

class TestLoadFile(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory and files
        self.test_dir = Path(tempfile.mkdtemp())
        self.temp_file = Path(tempfile.mkstemp(suffix = "pyc", dir = self.test_dir)[1])

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_return_none(self):
        self.assertIsNone(file_access.load_file(Path(""), file_access.CompressionMethod.NONE))

    def test_return_file(self):
        self.assertIsNotNone(file_access.load_file(self.temp_file.resolve(), file_access.CompressionMethod.NONE))

if __name__ == '__main__':
    unittest.main()