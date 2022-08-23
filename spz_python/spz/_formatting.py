import itertools

import numpy as np

concat = itertools.chain.from_iterable


def _to_level(indices, pointers, level=0, start=0, stop=None):
    index, *indices = indices
    if stop is None:
        stop = len(index)
    if not indices:
        for _ in range(start, stop):
            yield level
            level += 1
        return
    ptrs, *pointers = pointers
    for start, stop in zip(ptrs[start:stop], ptrs[start + 1 : stop + 1]):
        levels = list(_to_level(indices, pointers, level, start, stop))
        yield [level, levels]
        maxlevel = level
        while levels:
            vals = [val for val in levels if not isinstance(val, list)]
            if vals:
                maxlevel = max(maxlevel, max(vals))
            levels = list(concat(val for val in levels if isinstance(val, list)))
        level = maxlevel + 1


def index_levels(self):
    levels = list(_to_level(self._indices, self._pointers))
    rv = []
    rv.append([x for x, _ in levels])
    for _ in range(len(self._pointers) - 1):
        levels = list(concat(y for _, y in levels))
        rv.append([x for x, _ in levels])
    rv.append(list(concat(y for _, y in levels)))
    return rv


def _to_group(indices, pointers, group=None, start=0, stop=None):
    index, *indices = indices
    if stop is None:
        stop = len(index)
    if not indices:
        for i in range(start, stop):
            if group is None:
                yield i
            else:
                yield group
        return
    ptrs, *pointers = pointers
    for i, (start, stop) in enumerate(zip(ptrs[start:stop], ptrs[start + 1 : stop + 1])):
        cur_group = i if group is None else group
        groups = list(_to_group(indices, pointers, cur_group, start, stop))
        yield [cur_group, groups]


def index_groups(self):
    groups = list(_to_group(self._indices, self._pointers))
    rv = []
    rv.append([x for x, _ in groups])
    for _ in range(len(self._pointers) - 1):
        groups = list(concat(y for _, y in groups))
        rv.append([x for x, _ in groups])
    rv.append(list(concat(y for _, y in groups)))
    return rv


def get_layout(self, *, squared=False):
    indices = self._indices
    pointers = self._pointers

    # Xs is easy
    index_widths = [2 + max(2, len(str(index.max()))) for index in indices]
    pointers_widths = [max(4, len(str(ptr.max()))) for ptr in pointers]
    index_widths = [2 + max(2, len(str(index.max()))) for index in indices]
    pointers_widths = [max(4, len(str(ptr.max()))) for ptr in pointers]
    widths = [0]
    for ws in zip(index_widths[:-1], pointers_widths):
        widths.extend(ws)
    xoffsets = np.cumsum(widths)

    # Now we need to determins Ys.  Get initial guesses.
    groups = index_groups(self)
    yoffsets = [list(np.arange(len(index)) * 3) for index in indices[:-2]]
    last_in_group = np.diff(groups[-1])[pointers[-1][:-2]].astype(bool)
    diffed = np.diff(pointers[-1])[:-1]
    yoffsets.append(
        np.pad((np.maximum(2, diffed + last_in_group) + 1 + squared).cumsum(), (1, 0)).tolist()
    )
    yoffsets.append(list(range(len(indices[-1]))))  # We handle constraints in the line above

    # Now update by matching level ys
    levels = index_levels(self)
    inplay = set(zip(concat(levels), concat(yoffsets)))
    for level in sorted({x for x, _ in inplay}):
        heights = sorted({y for x, y in inplay if x == level})
        if len(heights) == 1:
            continue
        max_height = max(heights)
        new_yoffsets = []
        for ys, cur_levels in zip(yoffsets, levels):
            if level not in cur_levels or ys[cur_levels.index(level)] == max_height:
                new_yoffsets.append(ys)
                continue
            diff = max_height - ys[cur_levels.index(level)]
            new_yoffsets.append([y if lvl < level else y + diff for y, lvl in zip(ys, cur_levels)])
        yoffsets = new_yoffsets
        inplay = set(zip(concat(levels), concat(yoffsets)))

    return index_widths, pointers_widths, xoffsets, yoffsets


def autoexpand(func):
    """Make the canvas larger if there is an IndexError"""

    def inner(canvas, *args, **kwargs):
        for _ in range(1000):
            try:
                return func(canvas, *args, **kwargs)
            except IndexError:
                for row in canvas:
                    row.append(" ")
                canvas.append([" "] * len(canvas[0]))

    return inner


@autoexpand
def draw_box(canvas, x, y, width, val, *, dashed=False, square=False, cap_right=False):
    h = "~" if dashed else "-"
    v = "┊" if dashed else "|"
    canvas[y][x] = "+" if square else "."
    canvas[y + 1][x] = v
    canvas[y + 2][x] = "+" if square else "`"
    canvas[y][x + width - 1] = "+" if square else "."
    canvas[y + 1][x + width - 1] = v
    canvas[y + 2][x + width - 1] = "+" if square else "'"
    for w in range(1, width - 1):
        canvas[y][x + w] = h
        canvas[y + 2][x + w] = h
    sval = str(val).center(width - 2)
    for c, w in zip(sval, range(1, width - 1)):
        canvas[y + 1][x + w] = c
    if cap_right:
        canvas[y + 1][x + width] = ")"


@autoexpand
def draw_final_boxes(canvas, x, ys, width, index, *, square=False):
    prev_y = -999
    for y, idx in zip(ys, index):
        if y - prev_y > 2:
            canvas[y][x] = "+" if square else "."
            for w in range(1, width - 1):
                canvas[y][x + w] = "-"
            canvas[y][x + width - 1] = "+" if square else "."
            # Close previous box
            if prev_y >= 0:
                canvas[prev_y + 2][x] = "+" if square else "`"
                for w in range(1, width - 1):
                    canvas[prev_y + 2][x + w] = "-"
                canvas[prev_y + 2][x + width - 1] = "+" if square else "'"
        elif y - prev_y == 2:
            # Connected, but separate by horizontal line
            canvas[y][x] = "|"
            for w in range(1, width - 1):
                canvas[y][x + w] = "-"
            canvas[y][x + width - 1] = "|"
        canvas[y + 1][x] = "|"
        sval = str(idx).center(width - 2)
        for c, w in zip(sval, range(1, width - 1)):
            canvas[y + 1][x + w] = c
        canvas[y + 1][x + width - 1] = "|"
        prev_y = y
    # Close last box
    canvas[prev_y + 2][x] = "+" if square else "`"
    for w in range(1, width - 1):
        canvas[prev_y + 2][x + w] = "-"
    canvas[prev_y + 2][x + width - 1] = "+" if square else "'"


@autoexpand
def draw_line(canvas, x, y, w, *, dashed=False, double=False, chain=False):
    c = "~" if dashed else "-"
    if double:
        c = "="
    for i in range(w):
        if dashed and double and i % 2:
            continue
        if chain:
            c = "=" if i % 2 else "-"
        canvas[y][x + i] = c


@autoexpand
def draw_bendy_line(canvas, x, y1, y2, w, *, dashed=False, squared=False):
    h = "~" if dashed else "-"
    v = "┊" if dashed else "|"
    for i in range(w - 2, w):
        canvas[y2][x + i] = h
    for i in range(y1 + 1, y2):
        canvas[i][x + w - 3] = v
    canvas[y1][x + w - 3] = "+"
    canvas[y2][x + w - 3] = "+" if squared else "`"


@autoexpand
def draw_pointer(canvas, x, y, w, ptr, *, center_right=False, skip_if_nonempty=False):
    if center_right:
        sptr = str(ptr).center(w + 1)[:-1]
    else:
        sptr = str(ptr).center(w)
    if skip_if_nonempty:
        for i in range(len(sptr)):
            if canvas[y][x + i] != " ":
                return
    for i, c in enumerate(sptr):
        canvas[y][x + i] = c


def to_text(self, *, squared=False):
    indices = self._indices
    pointers = self._pointers
    index_widths, pointers_widths, xoffsets, yoffsets = get_layout(self, squared=squared)
    # Doesn't need to be perfect: we'll expand the canvas as needed
    xmax = max(xoffsets) + 1
    ymax = max(concat(yoffsets)) + 1
    canvas = [[" "] * xmax for _ in range(ymax)]
    # Draw boxes of indices
    for i, (width, x, ys, index, ptrs) in enumerate(
        zip(index_widths[:-1], xoffsets[::2], yoffsets, indices[:-1], pointers)
    ):
        dashed = self.indices[i] is None
        for y, idx, start, stop in zip(ys, index, ptrs[:-1], ptrs[1:]):
            cap_right = start == stop
            draw_box(canvas, x, y, width, idx, dashed=dashed, square=squared, cap_right=cap_right)
    draw_final_boxes(
        canvas, xoffsets[-1], yoffsets[-1], index_widths[-1], indices[-1], square=squared
    )

    # Draw lines for pointers
    for i, (x, w) in enumerate(zip(xoffsets[1::2], pointers_widths)):
        dashed = self.pointers[i] is None
        prev_start = 0
        for j, (yinit, start, stop) in enumerate(
            zip(yoffsets[i], pointers[i][:-1], pointers[i][1:])
        ):
            if j == 0 or prev_start == start:
                draw_pointer(canvas, x, yinit, w, start, center_right=True, skip_if_nonempty=True)
            for y in yoffsets[i + 1][start:stop]:
                if y == yinit:
                    draw_line(canvas, x, y + 1, w, dashed=dashed)
                else:
                    draw_bendy_line(canvas, x, yinit + 1, y + 1, w, dashed=dashed, squared=squared)
            draw_pointer(canvas, x, y + 2, w, stop, center_right=True)
            prev_start = start
        if start == stop:
            draw_pointer(
                canvas,
                xoffsets[2 * i + 1],
                yoffsets[i][-1] + 2,
                w,
                start,
                center_right=True,
                skip_if_nonempty=True,
            )

    # Draw header
    header = [[" "] * xmax for _ in range(4)]
    for i, (x, w, index) in enumerate(zip(xoffsets[::2], index_widths, self.indices)):
        draw_box(header, x, 0, w + 1, f"i{i} ", square=squared, dashed=index is None)
    for i, (x, w) in enumerate(zip(xoffsets[1::2], pointers_widths)):
        draw_pointer(header, x + 1, 1, w - 2, f"p{i}", center_right=False)
    draw_line(header, 0, 3, len("".join(header[0]).rstrip()), double=True)

    # Strip extra white space
    hrows = ["".join(row).rstrip() for row in header]
    while hrows and not hrows[-1]:
        hrows.pop()
    rows = ["".join(row).rstrip() for row in canvas]
    while rows and not rows[-1]:
        rows.pop()
    return "\n".join(hrows + rows)


def to_svg(self):
    from sphinxcontrib.svgbob._svgbob import to_svg as _to_svg

    text = to_text(self)
    return _to_svg(text)
