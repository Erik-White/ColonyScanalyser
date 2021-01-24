from typing import Any, Union, Tuple, List
from abc import ABC, abstractmethod
from numpy import ndarray
from skimage.transform._geometric import GeometricTransform, SimilarityTransform


class AlignTransform(ABC):
    """
    An abstract class to provide image alignment
    """
    def __init__(self, image_ref: ndarray, transform_model: GeometricTransform):
        """
        Initialise a new instance of the AlignTransform

        :param image_ref: a reference image to align with
        :param transform_model: a GeometricTransform type used when warping images to match image_ref
        """
        self.image_ref = image_ref
        self.transform_model = transform_model

    @property
    @abstractmethod
    def image_ref(self) -> Any:
        """
        Store the reference image as its descriptors, or similar
        """
        raise NotImplementedError("This property must be implemented in a derived class")

    @image_ref.setter
    @abstractmethod
    def image_ref(self, val: ndarray):
        """
        Store the reference image as its descriptors, or similar
        """
        raise NotImplementedError("This property must be implemented in a derived class")

    @property
    def transform_model(self) -> GeometricTransform:
        return self._transform_model

    @transform_model.setter
    def transform_model(self, val: GeometricTransform):
        self._transform_model = val

    @abstractmethod
    def align(self, image: ndarray, precise: bool = True, **kwargs) -> ndarray:
        """
        Align an image with the current reference image

        :param image: an image to align
        :param precise: peform a second alignment pass, more accurate but much slower
        :param kwargs: keyword arguments
        :returns: an image aligned with image_ref
        """
        raise NotImplementedError("This property must be implemented in a derived class")

    @abstractmethod
    def align_transform(self, image: ndarray, **kwargs) -> GeometricTransform:
        """
        Calculate the transformation needed to align the image with the current reference image

        :param image: an image to align
        :param kwargs: keyword arguments
        :returns: a transformation that will align the image with image_ref
        """
        raise NotImplementedError("This property must be implemented in a derived class")


class DescriptorAlignTransform(AlignTransform):
    """
    Image alignment using a DescriptorExtractor
    """
    from skimage.feature import ORB
    from skimage.feature.util import DescriptorExtractor

    def __init__(
        self,
        image_ref: ndarray,
        transform_model: GeometricTransform = SimilarityTransform,
        descriptor_extractor_model: DescriptorExtractor = ORB,
        **kwargs
    ):
        """
        Initialise a new instance of the DescriptorAlignTransform

        The reference image is stored as its descriptors and keypoints as extracted by descriptor_extractor_model

        :param image_ref: a reference image to align with
        :param transform_model: a GeometricTransform type used when warping images to match image_ref
        :param descriptor_extractor_model: a DescriptorExtractor and FeatureDectector type used for image feature extraction
        :param kwargs: keyword arguments used when initialising descriptor_extractor_model
        """
        self.descriptor_extractor = descriptor_extractor_model(**kwargs)
        super().__init__(image_ref, transform_model)

    @property
    def descriptor_extractor(self) -> DescriptorExtractor:
        return self._descriptor_extractor

    @descriptor_extractor.setter
    def descriptor_extractor(self, val: DescriptorExtractor):
        self._descriptor_extractor = val

    @property
    def image_ref(self) -> Tuple[List, List]:
        """
        The reference image as its descriptors and keypoints
        """
        return self._image_ref_descriptors, self._image_ref_keypoints

    @image_ref.setter
    def image_ref(self, val: ndarray):
        """
        Store the reference image as its descriptors and keypoints
        """
        self._image_ref_descriptors, self._image_ref_keypoints = self._extract_keypoints(val)

    def align(self, image: ndarray, precise: bool = True, **kwargs) -> ndarray:
        """
        Align an image with the current reference image

        :param image: an image to align
        :param precise: peform a second alignment pass, more accurate but much slower
        :param kwargs: keyword arguments passed to skimage.feature.match_descriptors
        :returns: an image aligned with image_ref
        """
        from warnings import warn
        from skimage.transform import warp

        # Calulcate the transformation
        transform = self.align_transform(image, **kwargs)

        if precise:
            try:
                # Perform second pass to get a very accurate alignment
                image_aligned = warp(image.copy(), transform.inverse, order = 1, preserve_range = False)
                transform += self.align_transform(image_aligned, **kwargs)
            except RuntimeError:
                warn("Unable to perform second pass image alignment, no keypoints could be found.", RuntimeWarning)

        # Adjust the image using the calculated transform
        return warp(image, transform.inverse, order = 3, preserve_range = True)

    def align_transform(self, image: ndarray, **kwargs) -> GeometricTransform:
        """
        Calculate the transformation needed to align the image with the current reference image

        :param image: an image to align
        :param kwargs: keyword arguments passed to skimage.feature.match_descriptors
        :returns: a transformation that will align the image with image_ref
        """
        from numpy import flip
        from skimage.feature import match_descriptors
        from skimage.measure import ransac

        descriptors_ref, keypoints_ref = self.image_ref
        descriptors, keypoints = self._extract_keypoints(image)

        try:
            # Used matched features to filter keypoints
            matches = match_descriptors(descriptors_ref, descriptors, cross_check = True, **kwargs)
            matches_ref, matches = keypoints_ref[matches[:, 0]], keypoints[matches[:, 1]]

            # Robustly estimate transform model with RANSAC
            transform_robust, inliers = ransac(
                (matches_ref, matches),
                self.transform_model,
                min_samples = 8,
                residual_threshold = 0.8,
                max_trials = 1000
            )
        except (RuntimeError, IndexError) as err:
            raise RuntimeError(err, "No feature matches could be found between the two images")

        # The translation needs to be inverted
        return (self.transform_model(rotation = transform_robust.rotation)
                + self.transform_model(translation = -flip(transform_robust.translation)))

    def _extract_keypoints(self, image: ndarray) -> Tuple[ndarray, ndarray]:
        """
        Detect and extract descriptors and keypoints from an image

        :param image: the image to extract descriptors and keypoints
        :returns: a tuple of descriptor and keypoints
        """
        from numpy import asarray
        from skimage.color import rgb2gray
        from ..imaging import image_as_rgb

        # ORB can only handle 2D arrays
        if len(image.shape) > 2:
            image = rgb2gray(image_as_rgb(image))

        try:
            self.descriptor_extractor.detect_and_extract(image)
        except AttributeError:
            self.descriptor_extractor.detect(image)
            self.descriptor_extractor.extract(self.descriptor_extractor.keypoints)

        return asarray(self.descriptor_extractor.descriptors), asarray(self.descriptor_extractor.keypoints)


class FastFourierAlignTransform(AlignTransform):
    """
    Image alignment using FFT based image registration
    """
    @property
    def image_ref(self) -> Any:
        """
        Store the reference image as its descriptors, or similar
        """
        return self._image_ref

    @image_ref.setter
    def image_ref(self, val: ndarray):
        """
        Store the reference image as its descriptors, or similar
        """
        self._image_ref = val

    def align(self, image: ndarray, precise: bool = True, **kwargs) -> ndarray:
        """
        Align an image with the current reference image

        :param image: an image to align
        :param precise: peform a second alignment pass, more accurate but much slower
        :param kwargs: keyword arguments passed to imreg_dft.imreg.similarity
        :returns: an image aligned with image_ref
        """
        from imreg_dft import transform_img

        iterations = 2 if precise else 1
        _, transform = self._align_transform(self.image_ref, image, numiter = iterations, **kwargs)

        return transform_img(image, transform.scale, transform.rotation, transform.translation, bgval = 0)

    def align_transform(self, image: ndarray, **kwargs) -> GeometricTransform:
        """
        Calculate the transformation needed to align the image with the current reference image

        :param image: an image to align
        :param kwargs: keyword arguments passed to imreg_dft.imreg.similarity
        :returns: a transformation that will align the image with image_ref
        """
        _, transform = self._align_transform(self.image_ref, image, **kwargs)

        return self.transform_model(matrix = transform.params)

    @staticmethod
    def _align_transform(image_ref: ndarray, image: ndarray, **kwargs) -> Tuple[ndarray, SimilarityTransform]:
        """
        Calculate the transformation needed to align the image with the a reference image

        :param image_ref: a refernce image to align with
        :param image: an image to align with image_ref
        :param kwargs: keyword arguments passed to imreg_dft.imreg.similarity
        :returns: a transformation that will align the image with image_ref, and the aligned greyscale image
        """
        from imreg_dft import similarity
        from skimage.color import rgb2gray
        from ..imaging import image_as_rgb

        # imreg_dft.similarity can't handle colour images
        image_ref_gray = rgb2gray(image_as_rgb(image_ref))
        image_gray = rgb2gray(image_as_rgb(image))

        transform_params = similarity(image_ref_gray, image_gray, **kwargs)
        transform = SimilarityTransform(
            scale = transform_params["scale"],
            rotation = transform_params["angle"],
            translation = transform_params["tvec"]
        )

        return transform_params["timg"], transform


def transform_parameters_equal(
    align_transform: GeometricTransform,
    align_transform_compare: Union[GeometricTransform, ndarray],
    tolerance: float = 0.1
) -> bool:
    """
    Verify if GeometricTransform parameters are equal, within a specified relative tolerance.

    :param align_transform: A GeometricTransform
    :param align_transform_compare: A GeometricTransform or 3x3 transformation matrix
    :param tolerance: The maximum absolute tolerance allowed
    :returns: True if the transform parameters are equal within the tolerance value
    """
    from numpy import allclose

    # If a transformation matrix has been supplied, use it to create a new
    # instance of the same type as align_transform
    if isinstance(align_transform_compare, ndarray) and align_transform_compare.shape == (3, 3):
        transform_type = type(align_transform)
        align_transform_compare = transform_type(matrix = align_transform_compare)
    elif not isinstance(align_transform_compare, GeometricTransform):
        raise ValueError("The supplied type or transformation matrix is invalid")

    return allclose(
        align_transform.params,
        align_transform_compare.params,
        atol = tolerance,
        equal_nan = True
    )