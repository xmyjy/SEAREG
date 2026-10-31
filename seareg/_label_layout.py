"""Place labels near their current nodes using physical display distances."""
import numpy as np


def _leader_length(center, anchor, half_size):
    vector = anchor - center
    ratio = np.min(np.divide(half_size, np.abs(vector),
                             out=np.full(2, np.inf), where=vector != 0))
    return np.linalg.norm(vector) * max(0, 1 - ratio)


def _segment_intersects_boxes(start, end, boxes):
    """Test display-coordinate segments against rectangles, including edges."""
    delta = np.broadcast_to(end - start, boxes[:, :2].shape)
    parallel = delta == 0
    outside = parallel & ((start < boxes[:, :2]) | (start > boxes[:, 2:]))
    first = np.divide(boxes[:, :2] - start, delta,
                      out=np.full_like(delta, -np.inf), where=~parallel)
    second = np.divide(boxes[:, 2:] - start, delta,
                       out=np.full_like(delta, np.inf), where=~parallel)
    enter = np.maximum(np.minimum(first, second).max(axis=1), 0)
    leave = np.minimum(np.maximum(first, second).min(axis=1), 1)
    return ~outside.any(axis=1) & (enter <= leave)


def place_labels(ax, texts, anchors, *, expand=(1.05, 1.2), max_leader_mm=6):
    """Choose nearby text boxes without overlaps or leaders crossing text.

    Candidate boxes are generated from the current node positions and the
    actual rendered text sizes. The shortest feasible leaders are preferred.
    No previous figure positions, data-space offsets or randomness are used.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    points = ax.transData.transform(anchors)
    pixels_per_point = ax.figure.dpi / 72
    padding = pixels_per_point
    bounds = ax.get_window_extent(renderer).extents + np.array(
        [padding, padding, -padding, -padding])
    legend = ax.get_legend().get_window_extent(renderer).extents.copy()
    legend += np.array([-padding, -padding, padding, padding])
    candidates = []
    gaps = np.r_[[4, 6, 9, 12, 16],
                 np.arange(20, max_leader_mm * 72 / 25.4 + 1e-9, 4)]
    for text, anchor in zip(texts, points):
        bbox = text.get_window_extent(renderer)
        width, height = bbox.width, bbox.height
        half_size = np.array([width, height]) / 2
        choices, seen = [], set()
        # Keep a visible line outside the text, including at the closest gap.
        for gap in gaps * pixels_per_point:
            for offset in (-.4, -.2, 0, .2, .4):
                positions = (
                    (anchor[0] + offset * width, anchor[1] + height / 2 + gap),
                    (anchor[0] + offset * width, anchor[1] - height / 2 - gap),
                    (anchor[0] + width / 2 + gap, anchor[1] + offset * 5 * height),
                    (anchor[0] - width / 2 - gap, anchor[1] + offset * 5 * height),
                )
                for position in positions:
                    if position in seen:
                        continue
                    seen.add(position)
                    center = np.array(position)
                    box = np.r_[center - half_size, center + half_size]
                    if (box[0] < bounds[0] or box[1] < bounds[1]
                            or box[2] > bounds[2] or box[3] > bounds[3]):
                        continue
                    if (box[0] < legend[2] and box[2] > legend[0]
                            and box[1] < legend[3] and box[3] > legend[1]):
                        continue
                    inside = ((points[:, 0] > box[0] - padding)
                              & (points[:, 0] < box[2] + padding)
                              & (points[:, 1] > box[1] - padding)
                              & (points[:, 1] < box[3] + padding))
                    if inside.any():
                        continue
                    length = _leader_length(center, anchor, half_size)
                    if length > max_leader_mm * ax.figure.dpi / 25.4:
                        continue
                    expanded = np.r_[center - half_size * expand,
                                     center + half_size * expand]
                    distance = center - anchor
                    cost = length ** 2 + .01 * float(distance @ distance)
                    choices.append((center, expanded, cost))
        if not choices:
            raise ValueError(f"No nearby annotation placement for {text.get_text()}")
        candidates.append(choices)

    starts = np.r_[0, np.cumsum([len(group) for group in candidates])]
    rows, columns, values, lower, upper = [], [], [], [], []
    row = 0
    for start, stop in zip(starts[:-1], starts[1:]):
        for column in range(start, stop):
            rows.append(row)
            columns.append(column)
            values.append(1.)
        lower.append(1.)
        upper.append(1.)
        row += 1
    for left in range(len(texts)):
        for right in range(left + 1, len(texts)):
            right_boxes = np.array([choice[1] for choice in candidates[right]])
            right_centers = np.array([choice[0] for choice in candidates[right]])
            for li, (center, box, _) in enumerate(candidates[left]):
                conflict = ((box[0] < right_boxes[:, 2])
                            & (box[2] > right_boxes[:, 0])
                            & (box[1] < right_boxes[:, 3])
                            & (box[3] > right_boxes[:, 1]))
                conflict |= _segment_intersects_boxes(points[left], center, right_boxes)
                conflict |= _segment_intersects_boxes(
                    points[right], right_centers,
                    np.repeat(box[None, :], len(right_centers), axis=0))
                for ri in np.flatnonzero(conflict):
                    rows.extend((row, row))
                    columns.extend((starts[left] + li, starts[right] + ri))
                    values.extend((1., 1.))
                    lower.append(-np.inf)
                    upper.append(1.)
                    row += 1
    count = int(starts[-1])
    matrix = coo_matrix((values, (rows, columns)), shape=(row, count)).tocsc()
    costs = np.array([choice[2] for group in candidates for choice in group])
    costs += np.arange(count) * 1e-7
    result = milp(
        costs, integrality=np.ones(count), bounds=Bounds(0, 1),
        constraints=LinearConstraint(matrix, lower, upper),
        options={"time_limit": 15, "mip_rel_gap": .005},
    )
    if result.x is None:
        raise ValueError("No non-overlapping nearby annotation layout: " + result.message)
    for index, text in enumerate(texts):
        group = result.x[starts[index]:starts[index + 1]]
        if np.count_nonzero(group > .5) != 1:
            raise ValueError("Annotation layout did not produce an integral solution")
        center = candidates[index][int(np.argmax(group))][0]
        text.set_ha("center")
        text.set_va("center")
        text.set_position(ax.transData.inverted().transform(center))
