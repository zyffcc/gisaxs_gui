"""The steps of an Analyze run as a tree, like a task list: what is done, what runs, what each step found.

A newer analysis package sends ``@live`` events of kind ``step``: ``id`` (a node of a fixed tree), ``state``
(``running``, ``done``, ``failed``, ``skipped``), ``title`` (English, fixed per id), ``summary`` (one line of what the
step found, may be null) and ``detail`` (the name of a small file in the result's ``trace/`` folder with what the step
found, or null). The tree:

* ``import`` — ``import.centre``, ``import.mask``, ``import.read``, ``import.onset``;
* ``windows``; ``networks`` — one ``networks.<family>`` per network the analysis runs; ``compare``;
* ``joint`` — ``joint.pool``, ``joint.fit``; ``finish`` — ``finish.report``, ``finish.present``.

A step may be reported again (``windows`` once per network): the last report counts. A step not reported yet is
pending; a parent whose child runs runs too. An older package sends no such event: ``StepTree.reported`` stays False and
the page keeps its four step lines.

After the run GIMaP keeps the tree it was told in the result's ``trace/gimap_steps.json`` (``to_record``); a result
without that file but with a ``trace`` folder gets the tree rebuilt from ``analysis.json`` and the detail files there
(``from_files``); a result without either has no tree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

TREE = (("import", ("import.centre", "import.mask", "import.read", "import.onset")), ("windows", ()),
        ("networks", ()), ("compare", ()), ("joint", ("joint.pool", "joint.fit")),
        ("finish", ("finish.report", "finish.present")))
"""(step, its children) in their order; the children of ``networks`` are the networks the analysis runs."""
STEP_TITLES = {
    "import": "Read the frames", "import.centre": "Find the beam centre", "import.mask": "Find the pixel mask",
    "import.read": "Read and bin the frames", "import.onset": "Find the start of the deposition",
    "windows": "Lay out the time windows", "networks": "The networks' solutions", "compare": "Compare the networks",
    "joint": "Joint fit of the whole scan", "joint.pool": "Prepare (every window tries its neighbours' solutions)",
    "joint.fit": "Iterations", "finish": "Write the result", "finish.report": "Report",
    "finish.present": "Pictures of the main screen"}
"""The title of each step (the package's own, in English; the page shows it in its language)."""
STATES = ("pending", "running", "done", "failed", "skipped")
NETWORK_PREFIX = "networks."
STEPS_RECORD = "gimap_steps.json"
"""What GIMaP keeps of the tree in the result's ``trace`` folder."""
TRACE_FOLDER = "trace"
DETAIL_OF = {"import.centre": "detector.json", "import.mask": "detector.json", "import.read": "reading.json",
             "import.onset": "onset.json", "windows": "windows.json", "compare": "compare.json",
             "joint": "joint.json", "joint.fit": "joint.json"}
"""The detail file a step writes (to rebuild the tree of a result that has its trace but not GIMaP's record)."""
PACKAGE_STEPS = {"import": "intake", "windows": "predict", "networks": "predict", "compare": "predict",
                 "joint": "predict", "finish": "present"}
"""The package step (``analysis.json`` ``steps``) a step of the tree belongs to."""


@dataclass
class StepNode:
    id: str
    title: str
    parent: str = ""
    state: str = "pending"
    summary: str = ""
    detail: str = ""
    """The file in the result's ``trace`` folder with what the step found ("" without one)."""

    @property
    def family(self) -> str:
        """The network of a ``networks.<family>`` step ("" for any other)."""
        return self.id[len(NETWORK_PREFIX):] if self.id.startswith(NETWORK_PREFIX) else ""


@dataclass
class StepTree:
    nodes: dict = field(default_factory=dict)
    """Step id -> StepNode, in the order of the tree (a parent before its children)."""
    reported: bool = False
    """A step event came (a newer package)."""

    def __post_init__(self) -> None:
        if not self.nodes:
            for parent, children in TREE:
                self.nodes[parent] = StepNode(parent, STEP_TITLES[parent])
                for child in children:
                    self.nodes[child] = StepNode(child, STEP_TITLES[child], parent)

    # -- taking events ------------------------------------------------------------------------------------------
    def feed(self, fields: dict) -> bool:
        """Take the fields of a ``step`` event; False when it names no step or no state the tree knows."""
        step_id, state = str(fields.get("id") or ""), str(fields.get("state") or "")
        if not step_id or state not in STATES:
            return False
        node = self._node(step_id, str(fields.get("title") or ""))
        summary, detail = fields.get("summary"), fields.get("detail")
        node.state = state
        node.summary = str(summary) if isinstance(summary, str) else ""
        node.detail = _file_name(detail) or (node.detail if state != "running" else "")
        parent = self.nodes.get(node.parent)
        if parent is not None and state == "running" and parent.state == "pending":
            parent.state = "running"
        self.reported = True
        return True

    def families(self, names) -> None:
        """The networks the analysis runs (the ``models`` event): their steps, pending until they start."""
        for name in names:
            if name:
                self._node(NETWORK_PREFIX + str(name), "")

    def _node(self, step_id: str, title: str) -> StepNode:
        node = self.nodes.get(step_id)
        if node is not None:
            return node
        parent = step_id.split(".", 1)[0] if "." in step_id else ""
        if parent and parent not in self.nodes:
            self.nodes[parent] = StepNode(parent, STEP_TITLES.get(parent, parent))
        name = step_id[len(NETWORK_PREFIX):] if step_id.startswith(NETWORK_PREFIX) else ""
        node = StepNode(step_id, name or STEP_TITLES.get(step_id) or title or step_id, parent)
        ordered = list(self.nodes.items())
        place = len(ordered)  # after the parent's last child, else at the end
        if parent:
            indices = [index for index, (key, item) in enumerate(ordered) if key == parent or item.parent == parent]
            place = indices[-1] + 1
        ordered.insert(place, (step_id, node))
        self.nodes = dict(ordered)
        return node

    # -- what the page shows ------------------------------------------------------------------------------------
    def ordered(self) -> list:
        return list(self.nodes.values())

    def children(self, step_id: str) -> list:
        return [node for node in self.nodes.values() if node.parent == step_id]

    def get(self, step_id: str) -> Optional[StepNode]:
        return self.nodes.get(step_id)

    def running(self) -> str:
        """The step at work: the last running step without a running child ("" when none runs)."""
        found = ""
        for node in self.nodes.values():
            if node.state == "running" and not any(child.state == "running" for child in self.children(node.id)):
                found = node.id
        return found

    def has_details(self) -> bool:
        return any(node.detail for node in self.nodes.values())

    # -- kept with the result -----------------------------------------------------------------------------------
    def to_record(self) -> dict:
        return {"schema": 1, "steps": [{"id": node.id, "parent": node.parent, "title": node.title,
                                        "state": node.state, "summary": node.summary, "detail": node.detail}
                                       for node in self.nodes.values()]}

    @classmethod
    def from_record(cls, record) -> Optional["StepTree"]:
        """The tree kept with a result (None when the record is not one)."""
        steps = record.get("steps") if isinstance(record, dict) else None
        if not isinstance(steps, list):
            return None
        tree = cls()
        for item in steps:
            if isinstance(item, dict) and item.get("id") and item.get("state") in STATES:
                node = tree._node(str(item["id"]), str(item.get("title") or ""))
                node.state = str(item["state"])
                node.summary = str(item.get("summary") or "")
                node.detail = _file_name(item.get("detail"))
        tree.reported = True
        return tree

    @classmethod
    def from_files(cls, analysis: dict, files) -> "StepTree":
        """The tree of a result that has its ``trace`` folder (``files``: the names in it) but not GIMaP's record:
        each step done, failed or skipped as ``analysis.json`` says its package step went, with the detail it wrote."""
        files, tree = set(files), cls()
        ran = {}
        for step in analysis.get("steps") or ():
            if isinstance(step, dict) and step.get("step"):
                name = str(step["step"]).split(" (", 1)[0]
                ran[name] = ran.get(name, True) and not step.get("returncode")
        tried = analysis.get("families_tried") if isinstance(analysis.get("families_tried"), dict) else {}
        tree.families(list(tried) or ([analysis["family"]] if analysis.get("family") else []))
        joint = analysis.get("joint_fit") if isinstance(analysis.get("joint_fit"), dict) else None
        for node in tree.nodes.values():
            top = node.parent or node.id
            went = ran.get("report" if node.id == "finish.report" else PACKAGE_STEPS.get(top, ""))
            node.state = "pending" if went is None else "done" if went else "failed"
            if top == "joint":
                node.state = "skipped" if joint is None else "done" if joint.get("done") else "failed"
            if DETAIL_OF.get(node.id) in files:
                node.detail = DETAIL_OF[node.id]
        tree.reported = True
        return tree


def _file_name(value) -> str:
    """A plain file name ("" for anything else: a path is never followed)."""
    name = value if isinstance(value, str) else ""
    return name if name and "/" not in name and "\\" not in name and name not in (".", "..") else ""


__all__ = ["DETAIL_OF", "NETWORK_PREFIX", "STATES", "STEPS_RECORD", "STEP_TITLES", "StepNode", "StepTree", "TRACE_FOLDER",
           "TREE"]
