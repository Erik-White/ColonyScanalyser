import pytest
import numpy as np

from scanlag.imaging import (crop_image,
                            cut_image_circle,
                            remove_background_mask,
                            watershed_separation,
                            standardise_labels_timeline,
                            get_labelled_centers,
                            replace_image_point_labels
                            )

image_ref = np.array(
    [[0, 0, 0, 0, 0, 0, 0, 1, 0],
    [1, 1, 0, 0, 1, 0, 0, 1, 0],
    [1, 1, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 0, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 0, 1, 0, 0],
    [0, 0, 1, 0, 1, 1, 1, 0, 1],
    [0, 1, 0, 0, 0, 0, 0, 0, 0]])

image_ref_segmented = image_ref.copy()
image_ref_segmented[1:3, 0:2] = 2
image_ref_segmented[0:2, 7] = 3
image_ref_segmented[6, 8] = 4

@pytest.fixture(params = [image_ref])
def image(request):
    yield request.param

@pytest.fixture(params = [image_ref_segmented])
def image_segmented(request):
    yield request.param

@pytest.fixture(params = [True, False])
def boolean(request):
    yield request.param

class TestCropImage():
    @pytest.fixture(params = [(1, 1), (2, 4), (1, 4)])
    def crop_shape(self, request):
        yield request.param

    @pytest.fixture(params = [(-2, 3), (0, 0, 0), (20, 3)])
    def crop_shape_invalid(self, request):
        yield request.param

    @pytest.fixture(params = [(2, 2), (5, 5), (6, 3)])
    def center(self, request):
        yield request.param

    def test_crop_shape(self, image, crop_shape):
        image_ref = np.array(
            [[0, 1, 1, 1],
            [1, 0, 1, 1]])

        result = crop_image(image, crop_shape)
        
        assert result.shape == crop_shape
        if crop_shape == (2, 4):
            assert (result == image_ref).all()

    def test_crop_center(self, image, crop_shape, center):
        image_ref = np.array(
            [[1, 1, 1, 1],
            [0, 1, 0, 1]])
        result = crop_image(image, crop_shape, center)
        
        assert result.shape == crop_shape
        if crop_shape == (2, 4) and center == (6, 3):
            assert (result == image_ref).all()

    def test_shape_invalid(self, image, crop_shape_invalid):
        with pytest.raises(ValueError):
            crop_image(image, crop_shape_invalid)

    def test_crop_outside(self, image, center):
        with pytest.raises(ValueError):
            print(crop_image(image, image.shape, center))


class TestCutImageCircle():
    def test_cut_circle(self, image, boolean):
        image_ref = image.copy()
        if not boolean:
            image_ref[0:2] = 0
            image_ref[1][4] = 1
            image_ref[-1] = 0
            image_ref[:, 0] = 0
            image_ref[:, -1] = 0
        else:
            image_ref[0:-1, 2:7] = 0
            image_ref[2:7, 1:-1] = 0
            
        result = cut_image_circle(image, inverse = boolean)
        
        assert result.shape == image_ref.shape
        assert (result == image_ref).all()

    def test_circle_with_params(self, image, boolean):
        if not boolean:
            image_ref = np.array(
                [[0, 1, 0],
                [1, 0, 1],
                [0, 1, 0]])
        else:
            image_ref = np.array(
                [[0, 0, 0],
                [0, 0, 0],
                [1, 0, 1]])

        result = cut_image_circle(image, center = (2, 4), radius = 1, inverse = boolean)
        
        assert result.shape == image_ref.shape
        assert (result == image_ref).all()
        
    def test_exceed_shape(self, image):
        with pytest.raises(ValueError):
            image_radius = image.shape[0] // 2
            cut_image_circle(image, radius = image_radius +1)

    def test_exceed_bounds(self, image):
        with pytest.raises(ValueError):
            cut_image_circle(image, center = (2, 4), radius = 3)


class TestRemoveBackgroundMask():
    @pytest.fixture(params = [image_segmented])
    def image_segmented_local(self, request):
        image_segmented_local = image_ref_segmented.copy()
        image_segmented_local[image_segmented_local == 1] = 255
        yield image_segmented_local

    @pytest.fixture(params = [image_segmented])
    def image_mask(self, request):
        image_mask = image_ref_segmented.copy()
        image_mask[5:8] = 0
        yield image_mask

    def test_remove_background(self, image_segmented_local, image_mask):
        image_ref = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        
        result = remove_background_mask(image_segmented_local, image_mask)

        assert result.shape == image_segmented_local.shape
        assert (result == image_ref).all()

    def test_size_mismatch(self):
        with pytest.raises(ValueError):
            remove_background_mask(
                np.ones((5, 5), dtype = np.uint8),
                np.ones((3, 5), dtype = np.uint8)
                )

    def test_image_empty(self, image_mask):
        with pytest.raises(ValueError):
            remove_background_mask(np.array([]), image_mask)

    def test_mask_empty(self, image_segmented_local):
        with pytest.raises(ValueError):
            remove_background_mask(image_segmented_local, np.array([]))


class TestWatershedSeparation():
    def test_watershed(self, image):
        result = watershed_separation(image)

        assert result.shape == image.shape
        assert len(np.unique(result)) == 3

    def test_empty(self):
        result = watershed_separation(np.array([]))

        assert result.size == 0
        assert len(np.unique(result)) == 0


class TestStandariseLabelsTimeline():
    @pytest.fixture(params = [image_segmented])
    def images_list(self, request):
        images_list = []

        # Loop through and alter labels so there are fewer repeats
        for i in range(1, 5, 1):
            image = image_ref_segmented.copy()
            for j in range(len(np.unique(image)), 1, -1):
                image[image == j] = j + i
            images_list.append(image)

        yield images_list

    @pytest.fixture(params = range(1, 1000, 350))
    def count_offset(self, request):
        yield request.param

    @pytest.fixture(params = [-1, 0.0, 0.5])
    def count_offset_invalid(self, request):
        yield request.param

    def test_unique(self, images_list, image_segmented, boolean, count_offset):
        result = standardise_labels_timeline(
            images_list,
            start_at_end = boolean,
            count_offset = count_offset,
            in_place = False
            )

        assert len(result) == len(images_list)
        assert len(np.unique(result)) == len(np.unique(image_segmented))

    def test_in_place(self, images_list, boolean):
        result = standardise_labels_timeline(images_list, in_place = boolean)

        assert len(result) == len(images_list)
        result_equal = all([np.array_equal(before, after) for before, after in zip(images_list, result)])

        if boolean:
            assert result_equal
        else:
            assert not result_equal

    def test_offset_invalid(self, images_list, boolean, count_offset_invalid):
        with pytest.raises(ValueError):
            standardise_labels_timeline(
                images_list,
                start_at_end = boolean,
                count_offset = count_offset_invalid
                )

class TestGetLabelledCenters():
    def test_labels(self, image_segmented):
        # Get the labels in the original image, excluding 0
        segmented_labels = np.unique(image_segmented)
        segmented_labels = segmented_labels[1:]

        result = get_labelled_centers(image_segmented)
        
        assert len(result) == len(segmented_labels)
        assert all([result[0] == label for result, label in zip(result, segmented_labels)])

    def test_centers(self, image_segmented):
        centers = [
            (4.217391304347826, 3.869565217391304),
            (1.5, 0.5),
            (0.5, 7.0),
            (6.0, 8.0)]

        result = get_labelled_centers(image_segmented)
        
        assert len(result) == len(centers)
        assert all([result[1] == center for result, center in zip(result, centers)])


class TestReplaceImagePointLabels():
    @pytest.fixture(params = [image_segmented], scope = "function")
    def labels(self, request):
        labels = []
        # Generate a new set of labels
        labels = get_labelled_centers(image_ref_segmented.copy())
        labels = [(label + 100, point) for label, point in labels]

        yield labels


    def test_replace(self, image_segmented, labels):
        result = replace_image_point_labels(image_segmented, labels)
        result_labels = get_labelled_centers(result)
        
        assert len(labels) == len(result_labels)
        assert all([before == after for before, after in zip(labels, result_labels)])

    def test_in_place(self, image_segmented, labels, boolean):
        result = replace_image_point_labels(image_segmented, labels, in_place = boolean)
        labels_before = get_labelled_centers(image_segmented)
        result_labels = get_labelled_centers(result)

        assert len(labels) == len(result_labels)
        result_equal = all([before == after for before, after in zip(labels_before, result_labels)])

        if boolean:
            assert result_equal
        else:
            assert not result_equal