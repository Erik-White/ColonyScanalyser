import pytest
from unittest.mock import patch, MagicMock
from numpy import array, ndarray, zeros, identity
from skimage.metrics import normalized_root_mse
from skimage.transform._geometric import GeometricTransform, EuclideanTransform, SimilarityTransform, AffineTransform

from colonyscanalyser.align import AlignTransform, DescriptorAlignTransform, transform_parameters_equal


@pytest.fixture(params = [EuclideanTransform, SimilarityTransform, AffineTransform])
def transform_type(request):
    yield request.param


@pytest.fixture()
def image_ref(request):
    from skimage.data import coins

    yield coins()


@pytest.fixture(params = [0, 0.5, 1, -3])
def image_rotated(request, image_ref):
    from skimage.transform import rotate

    yield rotate(image_ref, request.param, order = 1)


@pytest.fixture(params = [(0, 0), (0, -2), (-2, -1), (-.34, 1.4), (1.5, -2)])
def image_translated(request, image_ref):
    from scipy.ndimage.interpolation import shift

    yield shift(image_ref, request.param, order = 0)


@patch.object(AlignTransform, "__abstractmethods__", set())
class TestAlignTransform:
    def test_init(self):
        with pytest.raises(NotImplementedError):
            AlignTransform(array([]), None)

    def test_transform_model(self, transform_type):
        # image_ref is not implemented
        AlignTransform.image_ref = MagicMock()
        instance = AlignTransform(array([]), transform_type)

        assert instance.transform_model == transform_type
        instance.transform_model = None
        assert instance.transform_model is None


class TestDescriptorAlignTransform:
    from skimage.feature import ORB

    # Only ORB currently supports both feature detection and extraction
    @pytest.fixture(params = [ORB])
    def extractor_type(self, request):
        yield request.param

    @pytest.fixture
    def instance(self, image_ref, extractor_type):
        yield DescriptorAlignTransform(image_ref, EuclideanTransform, extractor_type)

    @patch.object(DescriptorAlignTransform, "_extract_keypoints")
    def test_init(self, patch_extract_keypoints, transform_type, extractor_type):
        image_keypoints = ([0], [0])
        patch_extract_keypoints.return_value = image_keypoints
        instance = DescriptorAlignTransform(zeros((2, 2)), transform_type, extractor_type)

        assert instance.image_ref == image_keypoints
        assert instance.transform_model == transform_type
        assert isinstance(instance.descriptor_extractor, extractor_type)

    @pytest.mark.parametrize("keypoints_count", [0, 10, 100, 1000])
    def test_image_ref(self, keypoints_count, image_ref, extractor_type):
        instance = DescriptorAlignTransform(image_ref, GeometricTransform, extractor_type, n_keypoints = keypoints_count)

        assert len(instance.image_ref[0]) == keypoints_count
        assert isinstance(instance.descriptor_extractor, extractor_type)

    def test_image_align_rotated(self, instance, image_ref, image_rotated):
        result = instance.align(image_rotated, precise = True)

        assert result.shape == image_rotated.shape == image_ref.shape
        assert round(normalized_root_mse(image_ref, result), 5) < 0.9962

    def test_image_align_translated(self, instance, image_ref, image_translated):
        result = instance.align(image_translated, precise = True)

        assert result.shape == image_translated.shape == image_ref.shape
        assert round(normalized_root_mse(image_ref, result), 5) < 0.15

    def test_image_align_features(self, instance):

        with pytest.raises(RuntimeError):
            instance.align(zeros((2, 2)), precise = False)

    @patch.object(DescriptorAlignTransform, "align_transform")
    @pytest.mark.parametrize("precise", [True, False])
    def test_image_align_precise(self, mock_align_transform, precise, instance, image_ref, image_translated):
        mock_align_transform.side_effect = [EuclideanTransform(), RuntimeError()]

        if precise:
            with pytest.warns(RuntimeWarning):
                instance.align(image_translated, precise = precise)
        else:
            assert instance.align(image_translated, precise = precise).shape == image_ref.shape

    @patch("skimage.feature.match_descriptors")
    def test_image_align_transform(self, mock_match_descriptors, instance, image_translated):
        mock_match_descriptors.side_effect = IndexError()

        with pytest.raises(RuntimeError):
            instance.align_transform(image_translated)

    def test_extract_keypoints(self, instance, image_ref):
        from skimage.color import gray2rgb

        result = instance._extract_keypoints(gray2rgb(image_ref))

        assert len(result) == 2
        assert len(result[0]) == 500
        assert len(result[1]) == 500


class TestTransformParametersEqual:
    @pytest.fixture
    def transform(self, request):
        yield EuclideanTransform(matrix = identity(3))

    @pytest.mark.parametrize("transform_type", [GeometricTransform, ndarray])
    def test_input_type(self, transform, transform_type):
        from copy import copy

        if transform_type == ndarray:
            transform_compare = transform.params
        else:
            transform_compare = copy(transform)

        assert transform_parameters_equal(transform, transform_compare, 0)

    @pytest.mark.parametrize("tolerance", [0, 0.1, 0.5, 1])
    def test_tolerance(self, transform, tolerance):
        from copy import deepcopy

        transform_compare = deepcopy(transform)
        transform_compare.params += tolerance
        print(transform_compare.params)

        assert transform_parameters_equal(transform, transform_compare, tolerance)

        transform_compare.params += 0.001

        assert not transform_parameters_equal(transform, transform_compare, tolerance)