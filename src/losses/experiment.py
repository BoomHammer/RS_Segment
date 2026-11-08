"""Loss and label-expansion utilities for E08, E09 and E15."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from models.input_adapter import sanitize_batch


def _growth_domain(batch):
    domain = batch["valid_mask"].bool()
    if "core_mask" in batch:
        domain &= batch["core_mask"].bool()
    if "supervision_split_mask" in batch:
        domain &= batch["supervision_split_mask"].bool()
    return domain


@torch.no_grad()
def feature_region_labels(batch, *, distance_threshold=0.75, max_radius=8):
    """Grow train-owned points through connected, similar input pixels."""

    # Dataset batches deliberately expose only raw masks. Reuse the model's
    # sanitization contract instead of expecting its derived value masks.
    clean = sanitize_batch(batch)
    dynamic = clean["dynamic"]
    mask = (
        clean["dynamic_value_mask"] & clean["dynamic_time_mask"][:, :, None, None, None]
    )
    dynamic_mean = (dynamic * mask).sum(1) / mask.sum(1).clamp_min(1)
    features = torch.cat((dynamic_mean, clean["static"]), dim=1).float()
    scale = features.flatten(2).std(2, keepdim=True).clamp_min(1e-3)[..., None]
    features = features / scale
    labels = torch.zeros_like(batch["ground_truth"])
    occupied = torch.zeros_like(labels, dtype=torch.bool)
    domain = _growth_domain(batch)
    seeds = batch["ground_truth_mask"].bool() & domain
    for sample, row, column in seeds.nonzero(as_tuple=False).tolist():
        label = batch["ground_truth"][sample, row, column]
        distance = (
            features[sample] - features[sample, :, row, column, None, None]
        ).pow(2)
        eligible = distance.mean(0).sqrt().le(distance_threshold) & domain[sample]
        region = torch.zeros_like(eligible)
        region[row, column] = True
        for _ in range(max_radius):
            region = F.max_pool2d(region[None, None].float(), 3, 1, 1)[0, 0].bool()
            region &= eligible
        region &= ~occupied[sample]
        labels[sample][region] = label
        occupied[sample] |= region
    # Measured points remain ground truth, not duplicated as weak supervision.
    occupied &= ~seeds
    labels.masked_fill_(~occupied, 0)
    return labels, occupied


@torch.no_grad()
def prediction_region_labels(probability, batch, *, threshold=0.95, max_iterations=64):
    """CRGNet 8-neighbour expansion from measured points (paper Eq. 3)."""

    predicted = probability.argmax(1) + 1
    confidence = probability.amax(1)
    domain = _growth_domain(batch)
    output = batch["ground_truth"].clone()
    grown = batch["ground_truth_mask"].bool() & domain
    for _ in range(max_iterations):
        previous = grown.clone()
        for code in range(1, probability.shape[1] + 1):
            source = grown & output.eq(code)
            adjacent = F.max_pool2d(source[:, None].float(), 3, 1, 1)[:, 0].bool()
            accept = (
                adjacent
                & ~grown
                & domain
                & predicted.eq(code)
                & confidence.ge(threshold)
            )
            output[accept] = code
            grown |= accept
        if torch.equal(previous, grown):
            break
    return output, grown


def _lovasz_gradient(sorted_foreground):
    count = len(sorted_foreground)
    total = sorted_foreground.sum()
    intersection = total - sorted_foreground.float().cumsum(0)
    union = total + (1 - sorted_foreground).float().cumsum(0)
    gradient = 1 - intersection / union.clamp_min(1)
    if count > 1:
        gradient[1:] = gradient[1:] - gradient[:-1].clone()
    return gradient


def _lovasz_softmax(probability, labels, mask):
    """Multiclass Lovasz-Softmax over expanded pixels (CRGNet Eq. 8)."""

    pixels = probability.movedim(1, -1)[mask]
    targets = labels[mask] - 1
    losses = []
    for class_index in range(probability.shape[1]):
        foreground = targets.eq(class_index).float()
        if not foreground.any():
            continue
        errors = (foreground - pixels[:, class_index]).abs()
        errors, order = errors.sort(descending=True)
        losses.append(torch.dot(errors, _lovasz_gradient(foreground[order])))
    return torch.stack(losses).mean() if losses else probability.sum() * 0


def crgnet_loss(outputs, batch, settings, *, epoch=0):
    """Sparse base CE, expanded CE and bidirectional probability consistency."""

    base = outputs["base_output"]["fine_probability"]
    expanded = outputs["expanded_output"]["fine_probability"]
    labels, mask = prediction_region_labels(
        base.detach(),
        batch,
        threshold=float(settings.get("confidence_threshold", 0.95)),
        max_iterations=int(settings.get("max_iterations", 64)),
    )
    # Expose only newly expanded pixels to the existing training audit metrics.
    generated_mask = mask & ~batch["ground_truth_mask"].bool()
    batch["weak_label"] = labels.masked_fill(~generated_mask, 0)
    batch["weak_label_mask"] = generated_mask
    valid = _growth_domain(batch)
    sparse_mask = batch["ground_truth_mask"].bool() & valid
    sparse_target = (batch["ground_truth"] - 1).masked_fill(~sparse_mask, -1)
    base_loss = F.nll_loss(base.clamp_min(1e-8).log(), sparse_target, ignore_index=-1)
    expanded_loss = _lovasz_softmax(expanded, labels, mask)
    consistency = ((base - expanded).pow(2).sum(1)[valid]).mean()
    self_training = base_loss.new_zeros(())
    batch["self_training_label"] = torch.zeros_like(batch["ground_truth"])
    batch["self_training_label_mask"] = torch.zeros_like(
        batch["ground_truth_mask"], dtype=torch.bool
    )
    if epoch >= int(settings.get("self_training_start_epoch", 20)):
        average = (base + expanded) / 2
        confidence, pseudo = average.detach().max(1)
        pseudo_mask = valid & confidence.ge(
            float(settings.get("self_training_threshold", 0.95))
        )
        if pseudo_mask.any():
            target = pseudo.masked_fill(~pseudo_mask, -1)
            batch["self_training_label"] = (pseudo + 1).masked_fill(~pseudo_mask, 0)
            batch["self_training_label_mask"] = pseudo_mask
            self_training = 0.5 * (
                F.nll_loss(base.clamp_min(1e-8).log(), target, ignore_index=-1)
                + F.nll_loss(expanded.clamp_min(1e-8).log(), target, ignore_index=-1)
            )
    total = (
        base_loss
        + expanded_loss
        + float(settings.get("consistency_weight", 1.0)) * consistency
        + float(settings.get("self_training_weight", 0.5)) * self_training
    )
    return {
        "loss": total,
        "ground_truth_loss": base_loss,
        "weak_label_loss": expanded_loss,
        "consistency_loss": consistency,
        "self_training_loss": self_training,
    }


def hssn_loss(outputs, batch, derived, *, gamma=2.0, triplet_weight=0.5):
    """Focal tree-min objective plus hierarchy-distance triplets."""

    scores = outputs["hssn_scores"].clamp(1e-6, 1 - 1e-6)
    counts = derived.get("level_counts") or [
        max(derived["fine_to_coarse"]) + 1,
        derived["num_classes"],
    ]
    parents = derived.get("level_parents", [derived["fine_to_coarse"]])
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    levels = batch.get("ground_truth_levels")
    if levels is None:
        leaf = batch["ground_truth"]
        targets = [leaf]
        for edge in reversed(parents):
            parent = torch.zeros_like(targets[-1])
            valid = targets[-1].gt(0)
            mapping = torch.as_tensor(edge, device=leaf.device)
            parent[valid] = mapping[targets[-1][valid] - 1] + 1
            targets.append(parent)
        levels = torch.stack(list(reversed(targets)), dim=1)
    domain = _growth_domain(batch)
    known = levels.gt(0) & domain[:, None]
    node_target = torch.zeros_like(scores)
    node_known = torch.zeros_like(scores, dtype=torch.bool)
    for depth, count in enumerate(counts):
        depth_known = known[:, depth]
        node_known[:, offsets[depth] : offsets[depth + 1]] = depth_known[:, None]
        for code in range(1, count + 1):
            node_target[:, offsets[depth] + code - 1] = levels[:, depth].eq(code)
    # Ancestor minima for positives and descendant maxima for negatives.
    coherent = scores.clone()
    paths = []
    for leaf in range(counts[-1]):
        path = [leaf]
        for edge in reversed(parents):
            path.append(edge[path[-1]])
        paths.append(list(reversed(path)))
    node_paths = {}
    for path in paths:
        indices = [offsets[d] + node for d, node in enumerate(path)]
        for position, index in enumerate(indices):
            node_paths.setdefault(index, indices[: position + 1])
    for index, ancestors in node_paths.items():
        descendants = [
            candidate for candidate, path in node_paths.items() if index in path
        ]
        positive = scores[:, ancestors].amin(1)
        negative = scores[:, descendants].amax(1)
        coherent[:, index] = torch.where(
            node_target[:, index].bool(), positive, negative
        )
    probability = torch.where(node_target.bool(), coherent, 1 - coherent).clamp_min(
        1e-6
    )
    focal = -((1 - probability).pow(gamma) * probability.log())
    tree_min = focal[node_known].mean()
    # Deterministic sparse triplets: same class is closest; another class is farther.
    embeddings = outputs["hssn_embeddings"].movedim(1, -1)
    leaf_mask = batch["ground_truth_mask"].bool() & domain
    vectors = embeddings[leaf_mask][:256]
    labels = (batch["ground_truth"][leaf_mask] - 1)[:256]
    triplet = tree_min.new_zeros(())
    terms = []
    if len(vectors) >= 3:
        normalized = F.normalize(vectors, dim=1)
        tree_distance = torch.zeros(
            counts[-1], counts[-1], device=labels.device, dtype=torch.float32
        )
        for first, first_path in enumerate(paths):
            for second, second_path in enumerate(paths):
                common = 0
                for left, right in zip(first_path, second_path, strict=True):
                    if left != right:
                        break
                    common += 1
                tree_distance[first, second] = 2 * (len(first_path) - common)
        hierarchy_height = max(len(paths[0]) - 1, 1)
        for index in range(len(vectors)):
            distances = tree_distance[labels[index], labels].clone()
            distances[index] = torch.inf
            positive_distance, positive = distances.min(0)
            distances[index] = -torch.inf
            negative_distance, negative = distances.max(0)
            if positive_distance < negative_distance:
                d_pos = 0.5 * (1 - (normalized[index] * normalized[positive]).sum())
                d_neg = 0.5 * (1 - (normalized[index] * normalized[negative]).sum())
                margin = 0.1 + 0.5 * (
                    (negative_distance - positive_distance) / (2 * hierarchy_height)
                )
                terms.append(F.relu(d_pos - d_neg + margin))
    if terms:
        triplet = torch.stack(terms).mean()
    return {
        "loss": tree_min + triplet_weight * triplet,
        "ground_truth_loss": tree_min,
        "weak_label_loss": tree_min.new_zeros(()),
        "tree_triplet_loss": triplet,
    }
