import numpy as np
import cv2
from img_utils import mat2gray

def transform_haar(mat, axes=[0, 1], depths=[8, 8], remove_zones={}, pad=False):
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
        half2_index[axis] = slice(int(mat.shape[axis] / 2), int(mat.shape[axis]))
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

        cv2.imwrite(f"axis{axis}depth{depth}additions.png", additions)
        cv2.imwrite(f"axis{axis}depth{depth}subtractions.png", subtractions)

        # Save Highpass and Lowpass sections into corresponding halves of the image
        mat[half1_index] = additions
        mat[half2_index] = subtractions

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
        depths=remaining_depths
    )

    # Remove zones, if passed in
    # We model our structure as a hypercube where each the first and second half of each image
    # along a given axis is a vertex in that axis, so for a 2D image, we have:
    # 00--01
    # |   |
    # 10--11
    for zone in remove_zones:
        ...

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
        half2_index[axis] = slice(int(mat.shape[axis] / 2), int(mat.shape[axis]))
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
        reconstructed_evens = (mat[half1_index] + mat[half2_index]) / (2**0.5)
        reconstructed_odds = (mat[half1_index] - mat[half2_index]) / (2**0.5)

        cv2.imwrite(f"axis{axis}depth{depth}evens-inverse.png", reconstructed_evens)
        cv2.imwrite(f"axis{axis}depth{depth}odds-inverse.png", reconstructed_odds)

        # Save Highpass and Lowpass sections into corresponding halves of the image
        mat[even_index] = reconstructed_evens
        mat[odd_index] = reconstructed_odds

        cv2.imwrite(f"axis{axis}depth{depth}-inverse.png", mat)

    return mat


im = cv2.imread("test.jpg").astype(np.double)#[:,:,0]
#cv2.imwrite("testgrey.jpg", im)
transform_haar(im, [0, 1], [1, 2])
cv2.imwrite("testout.jpg", im)
transform_haar_inverse(im, [0, 1], [1, 2])
cv2.imwrite("testoutreconstruct.jpg", im)
x=2





