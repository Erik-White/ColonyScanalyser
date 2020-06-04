import pytest
from unittest import mock
from datetime import datetime, timedelta
from pathlib import Path
from numpy import array

from colonyscanalyser.image_file import (
    ImageFile,
    ImageFileCollection
)


@pytest.fixture
def image(request):
    # A single pixel png image
    image_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    image_bytes += b"\x00\x00\x00\x0bIDAT\x08\x99c\xf8\x0f\x04\x00\t\xfb\x03\xfd\xe3U\xf2\x9c\x00\x00\x00\x00IEND\xaeB`\x82"

    return image_bytes


class TestImageFile:
    @pytest.fixture(params = [
        (datetime(2019, 4, 2, 14, 23), "img_20190402_1423.tif"),
        (datetime(1901, 12, 1, 23, 1), "1901-12-01-23:01.png"),
        (datetime(2020, 1, 31, 0, 59), "image 2020 01 31 00 59.gif")
    ])
    def timestamp_image(self, request):
        yield request.param

    @staticmethod
    def create_temp_file(tmp_path, file_name, suffix = "", file_data = b"1"):
        from tempfile import mkstemp

        _, temp_file = mkstemp(prefix = file_name, suffix = suffix, dir = tmp_path)

        # Write some data to the file so it is not empty
        with open(temp_file, "wb") as f:
            f.write(file_data)

        return temp_file

    @pytest.fixture(params = [True, False], autouse = False)
    def cache_image(self, request):
        yield request.param

    @pytest.mark.usefixtures("cache_image")
    class TestInitialize:
        def test_init(self, tmp_path, timestamp_image, cache_image, image):
            image_path = TestImageFile.create_temp_file(tmp_path, timestamp_image[1], suffix = "png", file_data = image)
            imagefile = ImageFile(image_path, cache_image = cache_image)

            assert str(imagefile.file_path) == str(image_path)
            assert imagefile.timestamp == imagefile.timestamp_initial
            assert imagefile.timestamp_elapsed == timedelta()
            assert imagefile.cache_image == cache_image

        def test_init_timestamp(self, tmp_path, timestamp_image, cache_image):
            timestamp, image_name = timestamp_image
            timestamp_diff = timedelta(hours = 1)
            image_path = TestImageFile.create_temp_file(tmp_path, image_name)
            imagefile = ImageFile(
                image_path,
                timestamp = timestamp,
                timestamp_initial = timestamp - timestamp_diff
            )

            assert imagefile.timestamp == timestamp
            assert imagefile.timestamp_initial == timestamp - timestamp_diff
            assert imagefile.timestamp_elapsed == timestamp_diff

        def test_enter_exit(self, tmp_path, timestamp_image, cache_image, image):
            image_path = TestImageFile.create_temp_file(tmp_path, timestamp_image[1], suffix = "png", file_data = image)
            imagefile = ImageFile(image_path, cache_image = cache_image)

            with imagefile as image_file:
                assert (image_file._ImageFile__image == array([[[255, 255, 255, 255]]])).all()
            if imagefile.cache_image:
                assert (imagefile._ImageFile__image == array([[[255, 255, 255, 255]]])).all()
            else:
                assert imagefile._ImageFile__image is None

    class TestProperties:
        @pytest.mark.parametrize("image_path", ["", Path(), "."])
        def test_filepath_missing(self, image_path):
            with pytest.raises(FileNotFoundError):
                ImageFile(image_path)

        def test_image(self, tmp_path, timestamp_image, image, cache_image):
            _, image_name = timestamp_image
            image_path = TestImageFile.create_temp_file(tmp_path, image_name, suffix = "png", file_data = image)
            imagefile = ImageFile(image_path, cache_image = cache_image)

            assert (imagefile.image == array([[[255, 255, 255, 255]]])).all()
            assert (imagefile.image_gray == array([[1.]])).all()

            if cache_image:
                assert (imagefile._ImageFile__image == array([[[255, 255, 255, 255]]])).all()
            else:
                assert imagefile._ImageFile__image is None

    class TestMethods:
        def test_timestamp_from_exif(self, tmp_path, timestamp_image):
            timestamp, image_name = timestamp_image
            image_path = TestImageFile.create_temp_file(tmp_path, image_name)
            imagefile = ImageFile(image_path)

            with pytest.raises(NotImplementedError):
                imagefile.timestamp_from_exif(image_path)

        def test_timestamp_from_string(self, tmp_path, timestamp_image):
            timestamp, image_name = timestamp_image
            image_path = TestImageFile.create_temp_file(tmp_path, image_name)
            imagefile = ImageFile(image_path)

            assert imagefile.timestamp_from_string(image_path) == timestamp

        def test_timestamp_from_string_invalid(self, tmp_path):
            image_path = TestImageFile.create_temp_file(tmp_path, "test_image_123456789")
            imagefile = ImageFile(image_path)

            with pytest.raises(ValueError):
                imagefile.timestamp_from_string(image_path, pattern = "")
                imagefile.timestamp_from_string("")
            assert imagefile.timestamp_from_string(imagefile.file_path.name) is None


class TestImageFileCollection:
    @staticmethod
    def ImageFileMock(file_path, timestamp):
        image_file = mock.Mock(spec = ImageFile)
        image_file.file_path = file_path
        image_file.timestamp = timestamp
        image_file.timestamp_initial = image_file.timestamp - timedelta(hours = 1)

        return image_file

    @pytest.fixture
    def image_files(self):
        image_files = list()
        timestamp = datetime.now()

        for i in range(10):
            image_files.append(self.ImageFileMock(str(i), timestamp + timedelta(hours = i)))

        return image_files

    class TestInitialize:
        def test_init(self):
            imagefiles = ImageFileCollection()

            assert imagefiles.items == list()

        def test_init_list(self, image_files):
            imagefiles = ImageFileCollection(image_files)

            assert imagefiles.items == image_files

    class TestProperties:
        def test_image_files_sorted(self, image_files):
            from random import sample

            image_files_shuffled = sample(image_files, len(image_files))
            imagefiles = ImageFileCollection(image_files_shuffled)

            assert imagefiles.items != image_files_shuffled
            assert imagefiles.items == image_files

        def test_image_file_count(self, image_files):
            imagefiles = ImageFileCollection(image_files)

            assert imagefiles.count == len(image_files)

        def test_file_paths(self, image_files):
            imagefiles = ImageFileCollection(image_files)

            assert len(imagefiles.file_paths) == len(image_files)
            assert imagefiles.file_paths == [image_file.file_path for image_file in image_files]

        def test_timestamps(self, image_files):
            imagefiles = ImageFileCollection(image_files)

            assert len(imagefiles.timestamps) == len(image_files)
            assert imagefiles.timestamps == [image_file.timestamp for image_file in image_files]

        def test_timestamps_initial(self, image_files):
            imagefiles = ImageFileCollection(image_files)

            assert len(imagefiles.timestamps_initial) == len(image_files)
            assert imagefiles.timestamps_initial == [image_file.timestamp_initial for image_file in image_files]

            timestamp_initial = datetime(1, 1, 1, 1, 1, 1)
            imagefiles.timestamps_initial = timestamp_initial

            assert imagefiles.timestamps_initial == [image_file.timestamp_initial for image_file in image_files]

        def test_timestamps_elapsed(self, image_files):
            imagefiles = ImageFileCollection(image_files)

            assert len(imagefiles.timestamps_elapsed) == len(image_files)
            assert imagefiles.timestamps_elapsed == [image_file.timestamp_elapsed for image_file in image_files]
            assert imagefiles.timestamps_elapsed_hours == [image_file.timestamp_elapsed_hours for image_file in image_files]
            assert (
                imagefiles.timestamps_elapsed_minutes == [image_file.timestamp_elapsed_minutes for image_file in image_files]
            )
            assert (
                imagefiles.timestamps_elapsed_seconds == [image_file.timestamp_elapsed_seconds for image_file in image_files]
            )

    class TestMethods:
        @pytest.fixture(params = [None, "10001010_00_00"])
        def timestamp_initial(self, request):
            yield request.param

        @mock.patch("colonyscanalyser.image_file.file_exists", return_value = True)
        def test_add_image_file(self, mock_file_exists, image_files):
            imagefiles = ImageFileCollection(image_files)
            image_file_first = imagefiles.items[0]
            new_image_file = imagefiles.add(
                file_path = "",
                timestamp = image_file_first.timestamp - timedelta(hours = 1),
                timestamp_initial = image_file_first.timestamp_initial - timedelta(hours = 1),
                cache_image = False
            )

            assert imagefiles.count == len(image_files) + 1
            assert new_image_file in imagefiles.items
            assert image_file_first != new_image_file
            assert imagefiles.items[0] == new_image_file

        @mock.patch("colonyscanalyser.image_file.file_exists", return_value = True)
        @mock.patch("colonyscanalyser.file_access.get_files_by_type", return_value = ["10001010_00_00"])
        def test_from_path(self, mock_get_files_by_type, mock_file_exists, timestamp_initial, tmp_path):
            results = ImageFileCollection.from_path(tmp_path, [])

            assert results.count == 1
            if timestamp_initial:
                assert results.timestamps_initial[0] == ImageFile.timestamp_from_string(timestamp_initial)
            else:
                assert results.timestamps_initial[0] == results.items[0].timestamp_initial

        def test_from_path_filenotfound(self, tmp_path):
            with pytest.raises(FileNotFoundError):
                ImageFileCollection.from_path(tmp_path, [])

        @mock.patch("colonyscanalyser.image_file.file_exists", return_value = True)
        @mock.patch("colonyscanalyser.file_access.get_files_by_type", return_value = ["test"])
        def test_from_path_ioerror(self, mock_get_files_by_type, mock_file_exists, tmp_path):
            with pytest.raises(IOError):
                ImageFileCollection.from_path(tmp_path, [])