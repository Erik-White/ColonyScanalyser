import pytest
import numpy as np

from colonyscanalyser.imaging import (
    mm_to_pixels,
    rgb_to_name,
    crop_image,
    cut_image_circle,
    get_image_circles,
    image_as_rgb,
    remove_background_mask,
    watershed_separation
)

image_ref = np.array([
    [0, 0, 0, 0, 0, 0, 0, 1, 0],
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
        image_ref = np.array([
            [0, 1, 1, 1],
            [1, 0, 1, 1]])

        result = crop_image(image, crop_shape)

        assert result.shape == crop_shape
        if crop_shape == (2, 4):
            assert (result == image_ref).all()

    def test_crop_center(self, image, crop_shape, center):
        image_ref = np.array([
            [1, 1, 1, 1],
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


class TestMMToPixels():
    @pytest.fixture(params = [100, 356, 95.6])
    def measurements(self, request):
        yield request.param

    @pytest.fixture(params = [150, 300, 2540])
    def dpi(self, request):
        yield request.param

    def test_dpi(self, measurements, dpi):
        result = mm_to_pixels(measurements, dpi)
        assert result == int(measurements * (dpi / 25.4))

    def test_ppmm(self, measurements, dpi):
        result = mm_to_pixels(measurements, dpi, pixels_per_mm = dpi)
        assert result == measurements * dpi

    @pytest.mark.parametrize("arg_invalid", [-1, 0, -25.4])
    def test_arg_invalid(self, arg_invalid):
        with pytest.raises(ValueError):
            mm_to_pixels(arg_invalid, arg_invalid, pixels_per_mm = arg_invalid)


class TestRGBToName():
    @pytest.fixture(params = ["html4", "css2", "css21", "css3"])
    def color_spec(self, request):
        yield request.param

    @pytest.fixture(params = [(255, 255, 255), (240, 240, 240), (255, 230, 240), (230, 230, 230), (255, 230, 220)])
    def colors(self, request):
        yield request.param

    def test_color_spec(self, color_spec):
        result = rgb_to_name((0, 0, 0), color_spec)
        assert result == "black"

    def test_colors(self, colors):
        result = rgb_to_name(colors, "css21")
        assert result == "white"


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
            image_ref = np.array([
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0]])
        else:
            image_ref = np.array([
                [0, 0, 0],
                [0, 0, 0],
                [1, 0, 1]])

        result = cut_image_circle(image, center = (2, 4), radius = 1, inverse = boolean)

        assert result.shape == image_ref.shape
        assert (result == image_ref).all()

    def test_exceed_shape(self, image):
        with pytest.raises(ValueError):
            image_radius = image.shape[0] // 2
            cut_image_circle(image, radius = image_radius + 1)

    def test_exceed_bounds(self, image):
        with pytest.raises(ValueError):
            cut_image_circle(image, center = (2, 4), radius = 3)


class TestGetImageCircles():
    @pytest.fixture
    def image_circle(self, request):
        # Create a 200x200 array with a donut shaped circle around the centre
        xx, yy = np.mgrid[:200, :200]
        circle = (xx - 100) ** 2 + (yy - 100) ** 2
        img = ((circle < (6400 + 60)) & (circle > (6400 - 60))).astype(np.uint8)
        img[img == circle] = 255

        yield img

    def test_get_circles(self, image_circle):
        result = get_image_circles(image_circle, 80, search_radius = 50)

        assert len(result) == 1
        assert result[0] == ((102, 102), 80)

    @pytest.mark.parametrize(
        "radius, count, search_radius, expected",
        [
            (80, 1, 10, ((102, 102), 80)),
            (80, None, 20, ((102, 102), 80)),
            (40, 4, 40, ((50, 46), 10)),
            (80, 1, 20, ((102, 102), 80)),
            (40, 4, 10, ((62, 62), 30))
        ])
    def test_get_circles_with_params(self, image_circle, radius, count, search_radius, expected):
        result = get_image_circles(
            image_circle,
            radius,
            circle_count = count,
            search_radius = search_radius
            )

        if count is None and expected == ((102, 102), 80):
            count = 1

        assert len(result) == count
        assert result[0] == expected

    def test_image_empty(self):
        with pytest.raises(ValueError):
            get_image_circles(np.array([]), 1)


class TestImageAsRGB():
    def test_grayscale(self, image):
        result = image_as_rgb(image)

        assert len(image.shape) == 2
        assert len(result.shape) == 3
        assert result.shape[-1] == 3

    def test_rgb(self, image):
        from numpy import empty

        image = empty(image.shape + (3, ), dtype = image.dtype)
        result = image_as_rgb(image)

        assert len(image.shape) == 3
        assert image.shape[-1] == 3
        assert len(result.shape) == 3
        assert result.shape[-1] == 3

    def test_rgba(self, image):
        from numpy import empty

        image = empty(image.shape + (4, ), dtype = image.dtype)
        result = image_as_rgb(image)

        assert len(image.shape) == 3
        assert image.shape[-1] == 4
        assert len(result.shape) == 3
        assert result.shape[-1] == 3


class TestRemoveBackgroundMask():
    @pytest.fixture(params = [image_segmented])
    def image_segmented_local(self, request):
        image_segmented_local = image_ref_segmented.copy()
        image_segmented_local[image_segmented_local > 1] = 255

        yield image_segmented_local

    def test_remove_background(self, image_segmented_local):
        image_ref = np.array([
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [1, 1, 0, 0, 1, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 1, 1, 0, 1],
            [0, 1, 0, 0, 0, 0, 0, 0, 0]
        ])

        result = remove_background_mask(image_segmented_local, smoothing = 0.1)

        assert result.shape == image_segmented_local.shape
        assert (result == image_ref).all()

    def test_image_blank(self):
        image_blank = np.zeros((3, 3))
        result = remove_background_mask(image_blank, image_blank > 0)

        assert (result == image_blank).all()

    def test_image_empty(self):
        with pytest.raises(ValueError):
            remove_background_mask(np.array([]))


class TestWatershedSeparation():
    def test_watershed(self, image):
        result = watershed_separation(image)

        assert result.shape == image.shape
        assert len(np.unique(result)) == 3

    def test_empty(self):
        result = watershed_separation(np.array([]))

        assert result.size == 0
        assert len(np.unique(result)) == 0