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


def apply_align_transform(
    image_file: ImageFile,
    align_model: Union[AlignTransform, GeometricTransform],
    **kwargs
) -> ImageFile:
    """
    Calculate a GeometricTransform for the ImageFile using the supplied AlignTransform,
    or use an existing transform

    :param image_file: an ImageFile to apply the transformation to
    :param align_model: a transform to use with image_file
    :param kwargs: keyword arguments to use with the AlignTransform.align_transform, if supplied
    :return: image_file with the transform applied
    """
    if isinstance(align_model, AlignTransform):
        image_file.alignment_transform = align_model.align_transform(image_file.image, **kwargs)
    else:
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

    Uses a binary search to recursively check the alignment of images with align_model

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

    return images[mid], mid