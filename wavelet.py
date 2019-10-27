import numpy as np
import cv2
from .img_utils import isKthBitOne, largestEvenLTE

DEBUG = False

def transform_haar(mat, axes=[0, 1], depths=[8, 8], remove_zones=[], copy=True):
    "NOTE: Destructive unless copy=True"
    # Pre-Processing
    assert len(axes) == len(depths)
    # Copy
    if copy:
        mat = mat.copy()
    # Check dtype
    if mat.dtype != np.double:
        mat = mat.astype(np.double, copy=False)

    # Base Case
    if not axes:
        return mat

    # Perform 1D Haar Wavelet Transform axis by axis
    for axis, depth in zip(axes, depths):
        # Make sure to skip if zero depth passed in.
        if depth <= 0:
            continue

        # Find important haar indices
        half1_index, half2_index, even_index, odd_index = haar_indices(mat.shape, axis)

        # Perform haar wavelet transform
        additions = (mat[odd_index] + mat[even_index]) / (2 ** 0.5)
        subtractions = (mat[odd_index] - mat[even_index]) / (2 ** 0.5)

        # Save Highpass and Lowpass sections into corresponding halves of the image
        mat[half1_index] = additions
        mat[half2_index] = subtractions

        if DEBUG:
            cv2.imwrite(f"axis{axis}depth{depth}additions.png", additions)
            cv2.imwrite(f"axis{axis}depth{depth}subtractions.png", subtractions)
            cv2.imwrite(f"axis{axis}depth{depth}.png", mat)

    # Isolate LP Area ("top-left")
    LP_area_index = LP_section(mat.shape, axes)

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
        remove_zones=remove_zones,
        copy=False  # Make sure not to make unecessary data copies in this recursive call
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
            axis_length = largestEvenLTE(mat.shape[axis])
            if isKthBitOne(zone, axis):  # If this bit is 1
                # Insert slice representing second half of this index
                zone_region_index[axis] = slice(int(axis_length / 2), axis_length)
            else:  # If this bit is 0
                # Insert slice representing first half of this index
                zone_region_index[axis] = slice(0, axis_length)
        zone_region_index = tuple(zone_region_index)  # Convert to tuple
        mat[zone_region_index] = 0

    # Return Result
    return mat

def transform_haar_inverse(mat, axes=[0, 1], depths=[8, 8], copy=True):
    """Inverse of above"""
    # Pre-Processing
    assert len(axes) == len(depths)
    # Copy
    if copy:
        mat = mat.copy()
    # Check dtype
    if mat.dtype != np.double:
        mat = mat.astype(np.double, copy=False)

    # Base Case
    if not axes:
        return mat

    # Isolate LP Area ("top-left")
    LP_area_index = LP_section(mat.shape, axes)

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
        depths=remaining_depths,
        copy=False  # Make sure not to make unecessary data copies in this recursive call
    )

    # Perform Inverse 1D Haar Wavelet Transform axis by axis
    for axis, depth in zip(axes, depths):
        # Make sure to skip if zero depth passed in.
        if depth <= 0:
            continue

        # Find important haar indices
        half1_index, half2_index, even_index, odd_index = haar_indices(mat.shape, axis)

        # Perform inverse haar wavelet transform
        reconstructed_odds = (mat[half1_index] + mat[half2_index]) / (2 ** 0.5)
        reconstructed_evens = (mat[half1_index] - mat[half2_index]) / (2 ** 0.5)

        # Save Highpass and Lowpass sections into corresponding halves of the image
        mat[even_index] = reconstructed_evens
        mat[odd_index] = reconstructed_odds

        if DEBUG:
            cv2.imwrite(f"axis{axis}depth{depth}evens-inverse.png", reconstructed_evens)
            cv2.imwrite(f"axis{axis}depth{depth}odds-inverse.png", reconstructed_odds)
            cv2.imwrite(f"axis{axis}depth{depth}-inverse.png", mat)

    return mat


def haar_indices(shape, axis):
    """Returns salient portions of the input matrix shape as tuples that can be used as slices in numpy ndarrays."""
    # Determine boundaries of this axis in the case that the length of this axis isn't even. If this is the case,
    # we do not include the end value in our calculation. Thus, our effective "axis_length" is the largest even
    # number less than the length of this axis.
    axis_length = largestEvenLTE(shape[axis])

    # Determine indices for the first and second halves of the image
    # which we will be saving our results to
    half1_index = null_slice(shape)
    half1_index[axis] = slice(0, int(axis_length / 2))
    half1_index = tuple(half1_index)
    half2_index = null_slice(shape)
    half2_index[axis] = slice(int(axis_length / 2), axis_length)
    half2_index = tuple(half2_index)

    # Determine indices for the even and odd halves of the image (along this axis)
    # to perform haar wavelet transform in one numpy step
    even_index = null_slice(shape)
    even_index[axis] = slice(0, axis_length, 2)
    even_index = tuple(even_index)
    odd_index = null_slice(shape)
    odd_index[axis] = slice(1, axis_length, 2)
    odd_index = tuple(odd_index)

    return half1_index, half2_index, even_index, odd_index

def LP_section(shape, axes):
    LP_area_index = null_slice(shape)
    for axis in axes:
        axis_length = largestEvenLTE(shape[axis])
        LP_area_index[axis] = slice(0, int(axis_length / 2))
    LP_area_index = tuple(LP_area_index)
    return LP_area_index

def null_slice(shape):
    return [slice(None) for _ in shape]

def decimate_haar(mat, axes, depths, remove_zones=[], copy=True):
    mat = transform_haar(mat, axes, depths, remove_zones, copy)
    mat = transform_haar_inverse(mat, axes, depths, copy)
    return mat


if __name__ == "__main__":
    # End-to-end sanity Test
    test_h = 5
    test_w = 13
    test_mat = np.arange(test_h*test_w).reshape((test_h, test_w))
    test_out = decimate_haar(
        mat=test_mat,
        axes=[0, 1],
        depths=[1, 2]
    )
    assert np.isclose(test_mat, test_out).all()

    # Visual Test
    im = cv2.imread("DSC01344.TIF").astype(np.double)
    im = decimate_haar(
        mat=im,
        axes=[0, 1],
        depths=[11, 11],
        remove_zones=[1, 3]
    )
    cv2.imwrite("testoutreconstruct.jpg", im)

