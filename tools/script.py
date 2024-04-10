import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch


def interpolate_points(points, new_num_pts):
    num_samples, num_pts, num_coords = points.shape

    # Calculate the pairwise distances for each set of points
    diffs = points[:, 1:] - points[:, :-1]
    dists = torch.sqrt(torch.sum(diffs ** 2, dim=2))

    # Cumulative sum to get the distance along the curve
    cum_dists = torch.cumsum(dists, dim=1)
    cum_dists = torch.cat([torch.zeros(num_samples, 1, device=points.device), cum_dists], dim=1)

    # Normalize the cumulative distances to [0, 1]
    max_dists = cum_dists[:, -1].unsqueeze(1)
    normalized_cum_dists = cum_dists / max_dists

    # Create the target cumulative distances for the interpolated points
    target_cum_dists = torch.linspace(0, 1, new_num_pts, device=points.device).unsqueeze(0).repeat(num_samples, 1)

    # Interpolate new points using the normalized cumulative distances
    new_points = torch.zeros(num_samples, new_num_pts, num_coords, device=points.device)

    # Set the start and end points
    new_points[:, 0, :] = points[:, 0, :]
    new_points[:, -1, :] = points[:, -1, :]

    for i in range(num_samples):
        for j in range(1, num_pts):
            # Get the mask of target points to be interpolated between the current pair of original points
            mask = (target_cum_dists[i] >= normalized_cum_dists[i, j - 1]) & (
                        target_cum_dists[i] < normalized_cum_dists[i, j])

            # Relative positions of the target points along the segment
            segment_dists = target_cum_dists[i, mask] - normalized_cum_dists[i, j - 1]
            segment_lengths = normalized_cum_dists[i, j] - normalized_cum_dists[i, j - 1]
            relative_positions = segment_dists / segment_lengths

            # Interpolate the points along the current segment
            new_points[i, mask] = (1 - relative_positions).unsqueeze(1) * points[
                i, j - 1] + relative_positions.unsqueeze(1) * points[i, j]

    return new_points


# Example usage:
num_samples = 10
num_pts = 5
num_coords = 3
points = torch.rand(num_samples, num_pts, num_coords)  # Example points tensor
new_num_pts = 10

interpolated_points = interpolate_points(points, new_num_pts)
print(interpolated_points)

for bs in range(num_samples):
    # Plot for points
    plt.figure(figsize=(12, 6))
    # Original points points
    plt.scatter(points[bs, :, 0], points[bs, :, 1], color='red', label='Original points')
    plt.plot(points[bs, :, 0], points[bs, :, 1], color='red', alpha=0.5)
    # Interpolated points points
    plt.scatter(interpolated_points[bs, :, 0], interpolated_points[bs, :, 1], color='blue', label='Interpolated points', marker='x')
    plt.plot(interpolated_points[bs, :, 0], interpolated_points[bs, :, 1], color='blue', alpha=0.5, linestyle='--')
    plt.title('points Points')
    plt.legend()
    plt.tight_layout()
    plt.show()