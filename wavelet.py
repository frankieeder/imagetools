import numpy as np
import cv2
from img_utils import isKthBitOne

DEBUG = False

def transform_haar(mat, axes=[0, 1], depths=[8, 8], remove_zones=[], pad=False):
    "NOTE: Destructive"
    assert len(axes) == len(depths)
    # Base Case
    if not axes:
        return mat

    # Perform Haar
    for axis, depth in zip(axes, depths):
        # Make sure to skip if zero depth passed in.
        if depth <= 0:
            continue

        # Determine indices for the first and second halves of the image
        # which we will be saving our results to
        half1_index = [slice(None) for _ in mat.shape]
        half1_index[axis] = slice(0, int(mat.shape[axis] / 2))
        half1_index = tuple(half1_index)
        half2_index = [slice(None) for _ in mat.shape]
        half2_index[axis] = slice(int(mat.shape[axis] / 2), mat.shape[axis])
        half2_index = tuple(half2_index)

        # Determine indices for the even and odd halves of the image (along this axis)
        # to perform haar wavelet transform in one numpy step
        even_index = [slice(None) for _ in mat.shape]
        even_index[axis] = slice(0, mat.shape[axis], 2)
        even_index = tuple(even_index)
        odd_index = [slice(None) for _ in mat.shape]
        odd_index[axis] = slice(1, mat.shape[axis], 2)
        odd_index = tuple(odd_index)

        # Perform haar wavelet transform
        additions = (mat[odd_index] + mat[even_index]) / (2**0.5)
        subtractions = (mat[odd_index] - mat[even_index]) / (2**0.5)



        # Save Highpass and Lowpass sections into corresponding halves of the image
        mat[half1_index] = additions
        mat[half2_index] = subtractions

        if DEBUG:
            cv2.imwrite(f"axis{axis}depth{depth}additions.png", additions)
            cv2.imwrite(f"axis{axis}depth{depth}subtractions.png", subtractions)
            cv2.imwrite(f"axis{axis}depth{depth}.png", mat)

    # Isolate LP Area ("top-left")
    LP_area_index = [slice(None) for _ in mat.shape]
    for axis in axes:
        LP_area_index[axis] = slice(0, int(mat.shape[axis] / 2))
    LP_area_index = tuple(LP_area_index)

    # Determine remaining axes to process and at what depths
    remaining_axes = []
    remaining_depths = []
    for c, ad in enumerate(zip(axes, depths)):
        axis, depth = ad
        if depth > 1:
            remaining_axes.append(axis)
            remaining_depths.append(depth-1)


    # Recursive call on LP Area
    mat[LP_area_index] = transform_haar(
        mat=mat[LP_area_index],
        axes=remaining_axes,
        depths=remaining_depths,
        remove_zones=remove_zones
    )

    # Remove zones, if passed in.
    # We model our structure as a hypercube where each the first and second half of each image
    # along a given axis is a vertex in that axis. We structure so that the least significant bit
    # corresponds to axis 0, the second least significant bit corresponds to axis 1, etc.
    #
    # So for a 2D image, we have:
    #
    # 00--10
    # |   |
    # 01--11
    #
    # or for 3D:
    #
    # 000 ---- 010
    #  |  \     |  \
    #  |  100 --|- 110
    #  |   |    |   |
    # 001 ---- 011  |
    #     \|       \|
    #     101 ---- 111
    #
    # We assume that we would only remove zones from the axes that we are operating on. As such, we will
    # only check the bits corresponding to those axes when looking at remove_zones.
    #

    for zone in remove_zones:
        # we want to make a slice isolating the zone given by the number in the above specifications
        zone_region_index = [slice(None) for _ in mat.shape]
        for axis in axes:
            if isKthBitOne(zone, axis):  # If this bit is 1
                # Insert slice representing second half of this index
                zone_region_index[axis] = slice(int(mat.shape[axis] / 2), mat.shape[axis])
            else:  # If this bit is 0
                # Insert slice representing first half of this index
                zone_region_index[axis] = slice(0, int(mat.shape[axis] / 2))
        zone_region_index = tuple(zone_region_index)  # Convert to tuple
        mat[zone_region_index] = 0


    # Return Result
    return mat


def transform_haar_inverse(mat, axes=[0, 1], depths=[8, 8]):
    """Inverse of above"""
    assert len(axes) == len(depths)

    # Base Case
    if not axes:
        return mat

    # Isolate LP Area ("top-left")
    LP_area_index = [slice(None) for _ in mat.shape]
    for axis in axes:
        LP_area_index[axis] = slice(0, int(mat.shape[axis] / 2))
    LP_area_index = tuple(LP_area_index)

    # Determine remaining axes to process and at what depths
    remaining_axes = []
    remaining_depths = []
    for c, ad in enumerate(zip(axes, depths)):
        axis, depth = ad
        if depth > 1:
            remaining_axes.append(axis)
            remaining_depths.append(depth - 1)

    # Recursive call on LP Area
    mat[LP_area_index] = transform_haar_inverse(
        mat=mat[LP_area_index],
        axes=remaining_axes,
        depths=remaining_depths
    )

    # Perform Inverse Haar
    for axis, depth in zip(axes, depths):
        # Make sure to skip if zero depth passed in.
        if depth <= 0:
            continue

        # Determine indices for the first and second halves of the image
        # which we will be saving to
        half1_index = [slice(None) for _ in mat.shape]
        half1_index[axis] = slice(0, int(mat.shape[axis] / 2))
        half1_index = tuple(half1_index)
        half2_index = [slice(None) for _ in mat.shape]
        half2_index[axis] = slice(int(mat.shape[axis] / 2), mat.shape[axis])
        half2_index = tuple(half2_index)

        # Determine indices for the even and odd halves of the image (along this axis)
        # to perform haar wavelet transform in one numpy step
        even_index = [slice(None) for _ in mat.shape]
        even_index[axis] = slice(0, mat.shape[axis], 2)
        even_index = tuple(even_index)
        odd_index = [slice(None) for _ in mat.shape]
        odd_index[axis] = slice(1, mat.shape[axis], 2)
        odd_index = tuple(odd_index)

        # Perform inverse haar wavelet transform
        reconstructed_odds = (mat[half1_index] + mat[half2_index]) / (2**0.5)
        reconstructed_evens = (mat[half1_index] - mat[half2_index]) / (2**0.5)



        # Save Highpass and Lowpass sections into corresponding halves of the image
        mat[even_index] = reconstructed_evens
        mat[odd_index] = reconstructed_odds

        if DEBUG:
            cv2.imwrite(f"axis{axis}depth{depth}evens-inverse.png", reconstructed_evens)
            cv2.imwrite(f"axis{axis}depth{depth}odds-inverse.png", reconstructed_odds)
            cv2.imwrite(f"axis{axis}depth{depth}-inverse.png", mat)

    return mat


def decimate_haar(mat, axes, depths, remove_zones=[]):
    transform_haar(mat, axes, depths, remove_zones)
    transform_haar_inverse(mat, axes, depths)

im = cv2.imread("test.jpg").astype(np.double)
decimate_haar(
    mat=im,
    axes=[0, 1],
    depths=[10, 10],
    remove_zones=[3]
)
cv2.imwrite("testoutreconstruct.jpg", im)





