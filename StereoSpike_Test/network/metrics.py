import numpy as np
import torch
import matplotlib.pyplot as plt


DISPARITY_MULTIPLIER = 7.0
FOCAL_LENGTH_X_BASELINE = {
    'indoor_flying': 19.941772,
}


def mask_dead_pixels(predicted, groundtruth):
    """
    Some pixels of the groundtruth have values of zero or NaN, which has no physical meaning. This function masks them.
    All such invalid values become 0, and valid ones remain as they are.
    """
    assert predicted.shape == groundtruth.shape, "input and target tensors do not have the same shape, can't apply " \
                                                 "the same mask to them ! " \
                                                 "Input is of shape {} and target of shape {}".format(predicted.shape,
                                                                                                      groundtruth.shape)
    masked_predicted = predicted.detach().clone()
    masked_groundtruth = groundtruth.detach().clone()

    mask = ~groundtruth.isnan()
    masked_groundtruth[mask == False] = 0

    return predicted, groundtruth


def depth_to_disparity(depth_maps):
    """
    Conversion from depth to disparity used in the paper "Learning an event sequence embedding for dense event-based
    deep stereo" (ICCV 2019)

    Original code available at https://github.com/tlkvstepan/event_stereo_ICCV2019
    """
    disparity_maps = DISPARITY_MULTIPLIER * FOCAL_LENGTH_X_BASELINE['indoor_flying'] / (depth_maps + 1e-15)
    return disparity_maps


def disparity_to_depth(disparity_map):
    depth_map = DISPARITY_MULTIPLIER * FOCAL_LENGTH_X_BASELINE['indoor_flying'] / (disparity_map + 1e-7)
    return depth_map


def lin_to_log_depths(depths_rect_lin, Dmax=80, alpha=3.7):
    """
    Applies normalized logarithm to the input depth maps. Log depth maps can represent large variations in a compact
    range, hence facilitating learning.
    Refer to the paper 'Learning Monocular Depth from Events' (3DV 2020) for more details.

    Basically,
    Dlin = Dmax * exp(-alpha * (1-Dlog))

    We only do this operation on elements that are different from 0 and 255, since those values (although present in the
     dataset) have no physical meaning and hinder learning

    With Dmax=10 and alpha=9.2, the minimum depth that can be predicted is of 0.001 meter.

    Also, predicted log depth should belong to [0; 1] interval.

    :param depths_rect_lin:  a tensor of shape [# of depth maps, W, H] containing depth maps at original linear scale
    :param Dmax: maximum expected depth
    :param alpha: parameter such that depth value of 0 maps to minimum observed depth. The bigger the alpha, the better
    the depth resolution.
    :return:  a tensor of shape [# of depth maps, W, H], but containing normalized log values instead
    """
    depths_rect_log = np.clip(depths_rect_lin, 0.0, Dmax)  # clip to maximum depth
    depths_rect_log = depths_rect_log / Dmax  # normalize
    depths_rect_log = 1.0 + np.log(depths_rect_log) / alpha  # take logarithm  # TODO changer alpha tq. lin_depth = Dmin --> log_depth = 0 |
    depths_rect_log = depths_rect_log.clip(0, 1.0)  # clip between 0 and 1

    return depths_rect_log


def log_to_lin_depths(depths_rect_log, Dmax=80, alpha=3.7):
    depths_rect_lin = Dmax * torch.exp(alpha * (depths_rect_log - torch.ones_like(depths_rect_log)))
    return depths_rect_lin


def MeanDepthError(predicted, groundtruth, cutoff_distance=float('inf')):
    """
    The Mean Depth Error (MDE) is commonly used as a metric to evaluate depth estimation algorithms on MVSEC dataset.
    It is computed only on non-NaN and non-zero values of the groundtruth depth map
    """
    mask = ~torch.isnan(groundtruth) & ~(groundtruth > cutoff_distance)  # True for valid pixels
    n = torch.count_nonzero(mask)  # number of valid pixels
    res = predicted - groundtruth  # calculate the residual
    res[mask == False] = 0  # invalid pixels: nan --> 0

    MDE = torch.sum(torch.abs(res[mask])) / n

    ## Original code
    # return MDE

    ## Mofication to return more information
    return MDE, n

# # Original code
def MeanDepthError_original(predicted, groundtruth, cutoff_distance=float('inf')):
    """
    The Mean Depth Error (MDE) is commonly used as a metric to evaluate depth estimation algorithms on MVSEC dataset.
    It is computed only on non-NaN and non-zero values of the groundtruth depth map
    """
    mask = ~torch.isnan(groundtruth) & ~(groundtruth > cutoff_distance)  # True for valid pixels
    n = torch.count_nonzero(mask)  # number of valid pixels
    res = predicted - groundtruth  # calculate the residual
    res[mask == False] = 0  # invalid pixels: nan --> 0

    MDE = torch.sum(torch.abs(res[mask])) / n

    return MDE


def OnePixelAccuracy(predicted_disp, groundtruth_disp):
    """
    The 1PA (One Pixel Accuracy) is the proportion of pixels where the predicted disparity is off by less than one pixel
    It was proposed in "Learning an event sequence embedding for dense deep event stereo" paper (ICCV2019).
    This piece of code is borrowed from their repository.

    Because our network predicts depth, we have to first convert its predicted depth maps to disparity using the same
    formula as the paper above.
    """
    locations_without_ground_truth = torch.isnan(groundtruth_disp) | groundtruth_disp.gt(255)
    more_than_n_pixels_absolute_difference = (predicted_disp - groundtruth_disp).abs().gt(1).float()
    pixelwise_n_pixels_error = more_than_n_pixels_absolute_difference.clone()
    pixelwise_n_pixels_error[locations_without_ground_truth] = 0.0
    more_than_n_pixels_absolute_difference_with_ground_truth = more_than_n_pixels_absolute_difference[
        ~locations_without_ground_truth]
    if more_than_n_pixels_absolute_difference_with_ground_truth.numel() == 0:
        percentage_of_pixels_with_error = 0.0
    else:
        percentage_of_pixels_with_error = \
            more_than_n_pixels_absolute_difference_with_ground_truth.mean(
            ).item() * 100
    return 100 - percentage_of_pixels_with_error
