from typing import Union, Tuple, Collection
from enum import Enum, auto
from skimage.transform._geometric import GeometricTransform
from .transform import AlignTransform
from ..image_file import ImageFile


class AlignStrategy(Enum):
    """
    The different methods utilised for aligning a collection of images

    quick:
        Locate start of image shift and compare to end. If both are similar, apply
        calculated alignment to all shifted images
    verify:
        Locate start of image shift and calculate alignment for all subsequent images
    complete:
        Calculate alignment for every single image
    none:
        Do not calculate alignment for any images
    """
    quick = auto()
    verify = auto(),
    complete = auto(),
    none = auto()


def calculate_transformation_strategy(
    images: Collection[ImageFile],
    strategy: AlignStrategy,
    transform_type: str = "euclidean",
    tolerance: float = 0.1,
    **kwargs
) -> Union[Union[AlignTransform, GeometricTransform], Collection[ImageFile]]:
    """
    Determine the AlignTransform and slice that it should be applied to in a
    collection of images, according the AlignStrategy.
    The first image in the collection is used as an alignment reference

    :param images: a collection of ImageFile instances to align
    :param strategy: the method used when processing images
    :param transform_type: a GeometricTransform, see skimage.transform._geometric.TRANSFORMS
    :param tolerance: the tolerance allowed in the image transformation matrix when using AlignStrategy.quick
    :param kwargs: keyword arguments passed to AlignTransform.align_transform
    :returns: An AlignTransform or precalculated GeometricTransform, depending on the selected AlignStrategy
    """
    from numpy import identity
    from skimage.transform._geometric import TRANSFORMS
    from .transform import FastFourierAlignTransform, transform_parameters_equal

    # Calculate alignment for all images by default (AlignStrategy.complete)
    shift_index = 0

    # There must be at least one other image to use as a reference
    if len(images) <= 1 or strategy == AlignStrategy.none:
        return images

    transform_type = transform_type.lower()
    if transform_type not in TRANSFORMS:
        raise ValueError(f"the transformation type {transform_type} is not implemented")
    transform_model = TRANSFORMS[transform_type]

    # Create the alignment model using the first image as a reference
    # The first image is cached in the model as keypoints
    align_model = FastFourierAlignTransform(images[0].image, transform_model)

    # Check the final image in the sequence in case alignment is not required
    image_file_final = images[-1]
    image_file_final.alignment_transform = align_model.align_transform(image_file_final.image)

    # If the first and last images are already aligned then there is nothing to do,
    # unless the strategy requires all images to be checked for alignment
    if not strategy == AlignStrategy.complete:
        # 3x3 identity matrix, equivalent to a stationary transformation matrix
        if transform_parameters_equal(image_file_final.alignment_transform, identity(3), tolerance):
            return align_model, list()

    if strategy == AlignStrategy.quick or AlignStrategy.verify:
        # Try to find the smallest number of images to align
        image_file_shift, shift_index = _locate_alignment_shift(images, align_model, tolerance = tolerance, **kwargs)
        # If the transformation at the start of the shift is very similar to the end,
        # apply the same transformation throughout. Otherwise use AlignStrategy.verify
        transforms_similar = transform_parameters_equal(
            image_file_shift.alignment_transform,
            image_file_final.alignment_transform,
            tolerance
        )
        if (strategy == AlignStrategy.quick and transforms_similar):
            align_model = image_file_final.alignment_transform

    return align_model, images[shift_index:]


def apply_align_transform(
    image_file: ImageFile,
    align_model: Union[AlignTransform, GeometricTransform],
    replace_existing: bool = False,
    **kwargs
) -> ImageFile:
    """
    Calculate a GeometricTransform for the ImageFile using the supplied AlignTransform,
    or use an existing transform

    :param image_file: an ImageFile to apply the transformation to
    :param align_model: a transform to use with image_file
    :param replace_existing: overwrite any existing image_file.alignment_transform
    :param kwargs: keyword arguments to use with the AlignTransform.align_transform, if supplied
    :return: image_file with the transform applied
    """
    if image_file.alignment_transform is None or replace_existing:
        if isinstance(align_model, AlignTransform):
            align_model = align_model.align_transform(image_file.image, **kwargs)

        image_file.alignment_transform = align_model

    return image_file


def _locate_alignment_shift(
    images: Collection[ImageFile],
    align_model: AlignTransform,
    tolerance: float = 0.1,
    **kwargs
) -> Tuple[ImageFile, int]:
    """
    Locate the first image in a collection where the image alignment shifted. The first image
    in the collection is used as the reference point.

    Uses a binary search to iteratively check the alignment of images with align_model

    :param images: a collection of ImageFile instances to check image alignment
    :param align_model: an AlignTransform used to calculate the image transformation matrix
    :param tolerance: the absolute difference allowed in the image transformation matrix
    :param kwargs: keyword arguments passed to AlignTransform.align_transform
    :returns: the ImageFile where alignment first shifted, and its index in the collection
    """
    from numpy import identity
    from .transform import transform_parameters_equal

    bottom = 0
    top = len(images) - 1

    while top - bottom > 1:
        mid = (top + bottom) // 2

        # Verify the image alignment
        image_file = images[mid]
        image_file.alignment_transform = align_model.align_transform(image_file.image, **kwargs)

        # 3x3 identity matrix, equivalent to a stationary transformation matrix
        if transform_parameters_equal(image_file.alignment_transform, identity(3), tolerance):
            image_file.align_transform = None
            bottom = mid
        else:
            top = mid
    mid += 1

    return images[mid], mid