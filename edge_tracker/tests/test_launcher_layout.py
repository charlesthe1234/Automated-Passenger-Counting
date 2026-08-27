"""Guard against Tkinter grid collisions in the launcher.

Two widgets placed in the same frame at the same (row, column) silently overlap:
whichever is gridded last is drawn on top and the other becomes invisible. There
is no error and no warning, and the machine running the launcher usually has no
spare display for an automated screenshot, so the mistake reaches the operator.

This checks the source statically instead.
"""

import ast
import unittest

from tests import EDGE_TRACKER_DIR

LAUNCHER = EDGE_TRACKER_DIR / "launcher_ui.py"


def _name_of(node):
    """Readable identifier for a widget or frame expression."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_name_of(node.value)}.{node.attr}"
    return None


def _keyword_int(call, name):
    for keyword in call.keywords:
        if keyword.arg == name and isinstance(keyword.value, ast.Constant):
            if isinstance(keyword.value.value, int):
                return keyword.value.value
    return None


def collect_grid_placements(source):
    """Map frame name -> list of (row, columns, description) for every .grid()."""
    tree = ast.parse(source)
    # Widget variable -> the frame it was constructed against.
    widget_parent = {}
    widget_label = {}

    def parent_of_construction(call):
        if not isinstance(call, ast.Call) or not call.args:
            return None
        return _name_of(call.args[0])

    def label_of_construction(call):
        kind = _name_of(call.func) or "widget"
        for keyword in call.keywords:
            if keyword.arg == "text" and isinstance(keyword.value, ast.Constant):
                return f"{kind}({keyword.value.value!r})"
        return kind

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            target = _name_of(node.targets[0]) if node.targets else None
            parent = parent_of_construction(node.value)
            if target and parent:
                widget_parent[target] = parent
                widget_label[target] = label_of_construction(node.value)

    placements = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "grid":
            continue
        row = _keyword_int(node, "row")
        column = _keyword_int(node, "column")
        if row is None or column is None:
            continue
        span = _keyword_int(node, "columnspan") or 1

        target = node.func.value
        if isinstance(target, ast.Call):
            # Chained form: ttk.Checkbutton(frame, ...).grid(...)
            frame = parent_of_construction(target)
            label = label_of_construction(target)
        else:
            name = _name_of(target)
            frame = widget_parent.get(name)
            label = widget_label.get(name, name)
        if frame is None:
            continue
        placements.setdefault(frame, []).append(
            (row, set(range(column, column + span)), f"{label} at row={row} column={column}")
        )
    return placements


class LauncherGridTests(unittest.TestCase):
    def test_no_two_widgets_share_a_grid_cell(self):
        placements = collect_grid_placements(LAUNCHER.read_text(encoding="utf-8"))
        self.assertTrue(placements, "no .grid() calls were parsed; the checker is broken")

        collisions = []
        for frame, entries in placements.items():
            for index, (row, columns, description) in enumerate(entries):
                for other_row, other_columns, other in entries[index + 1:]:
                    if row == other_row and columns & other_columns:
                        collisions.append(f"{frame}: {description}  overlaps  {other}")
        self.assertEqual(collisions, [], "overlapping launcher widgets:\n" + "\n".join(collisions))

    def test_the_experimental_checkbox_is_placed_exactly_once(self):
        placements = collect_grid_placements(LAUNCHER.read_text(encoding="utf-8"))
        matches = [
            description
            for entries in placements.values()
            for _row, _columns, description in entries
            if "3D Level Detection" in description
        ]
        self.assertEqual(len(matches), 2, f"expected the checkbox and its status label, got {matches}")

    def test_checker_detects_a_deliberate_collision(self):
        """The guard must actually fail when two widgets overlap."""
        source = (
            "import tkinter.ttk as ttk\n"
            "frame = ttk.Frame(None)\n"
            "ttk.Checkbutton(frame, text='a').grid(row=8, column=0, columnspan=2)\n"
            "ttk.Checkbutton(frame, text='b').grid(row=8, column=1)\n"
        )
        entries = collect_grid_placements(source)["frame"]
        overlap = entries[0][1] & entries[1][1]
        self.assertEqual(entries[0][0], entries[1][0])
        self.assertTrue(overlap)


if __name__ == "__main__":
    unittest.main()
