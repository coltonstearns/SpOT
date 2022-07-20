
import torch

# ----------------------------------------
# Point Cloud Sampling
# ----------------------------------------

def unique_idxs(x):
    unique, inverse = torch.unique(x, sorted=True, return_inverse=True)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    perm = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
    return perm


def group_first_k_values(values, batch, k):
    """
    Given a set of values and their batch-assignments, this operation selects and groups the first K values for
    each batch. If there are not K values in a batch, 0's are padded.
    Args:
        values (torch.tensor): 1D or 2D tensor of values
        batch (torch.LongTensor): 1D tensor of batch-idx assignments
        k (int or torch.tensor): number of values to group per-batch. If a tensor is given, the number of values is k.max(),
                                 and each batch will have a different limit of values (reflected via padding)

    Returns:
        grouped_values (torch.tensor): 2D tensor of size (num_batch, k)
        mask (torch.tensor): 2D tensor indicating which values are valid
    """
    # prelim attributes
    device, dims = values.device, len(values.size())
    num_batches = torch.unique(batch).size(0)
    if torch.is_tensor(k):
        k_max = torch.max(k).item()
    else:
        k_max = k

    # get first occurrence values in each batch
    batch_sorted, batch_sort_mapping = torch.sort(batch, stable=True)
    batch_initvals = torch.where(torch.cat([torch.ones(1).to(device), batch_sorted[1:] - batch_sorted[: -1]]))[0]

    # get the number of values assigned to each batch
    _, values_per_batch = torch.unique(batch_sorted, return_counts=True)
    values_per_batch = torch.clamp(values_per_batch, max=k).unsqueeze(1)

    # given first occurrence and values-per-batch, generate 2D array of indices
    inds = torch.arange(k_max).unsqueeze(0).expand(num_batches, k_max).to(device)
    mask = inds < values_per_batch
    inds = inds.clone() + batch_initvals.unsqueeze(1)
    inds[~mask] = 0

    # reformat to original, unsorted values
    values = values[batch_sort_mapping[inds.flatten()]].view(num_batches, k_max, -1)
    values[~mask] = 0
    if dims == 1:
        values = values.squeeze(-1)
    return values, mask

def merge_point2frame_across_batches(point2frameidx, point2batchidx):
    _, batch_counts = torch.unique(point2batchidx, return_counts=True)
    padded_point2frame, _ = group_first_k_values(point2frameidx, batch=point2batchidx, k=torch.max(batch_counts).item())
    frames_per_batch = torch.max(padded_point2frame, dim=1)[0] + 1
    frames_ptr = torch.cat([torch.zeros(1).to(frames_per_batch), torch.cumsum(frames_per_batch, dim=0)])
    point2frameidx = point2frameidx + frames_ptr[point2batchidx]
    return point2frameidx


