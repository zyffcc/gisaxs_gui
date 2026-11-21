"""The “Links and constraints” card of the Model step: particles do not overlap (contact), every link (``R_2 = 2*R_1``)
and every inequality constraint (``D_1 >= 2*R_1 + 3``), each with a button to remove it, and Add Constraint….

Links are made on a parameter's own line (its “=” button); inequalities here. Physics and Methods, section 9.15.
"""

from __future__ import annotations

from PyQt5.QtWidgets import QCheckBox, QFrame, QHBoxLayout, QInputDialog, QLabel, QPushButton, QToolButton

from src.gimap.app.presentation.i18n import tr, trf

from ...application.fit_uncertainty import Inequality, describe_links, parameter_name, with_inequalities

CONSTRAINT_HELP = ("A constraint keeps a fitted value on one side of an expression of others, e.g.\n"
                   "D_2 >= 2*R_2 + 3   or   R_2 <= 0.5*R_1\nNames: {names}.")
CONTACT_TIP = ("While both are fitted: the distance D of a paracrystal is at least the particles' diameter 2R "
               "(2(R + t) for core–shell spheres), and the hard-sphere radius R_HS at least R (+ t)")


def links_card(editor) -> QFrame:
    """The card (``editor``: the ``FitModelEditor``, whose model it shows and changes)."""
    card, layout = editor._card("fitLinks")
    title = QLabel(tr("Links and constraints"), card)
    title.setProperty("gimapRole", "strong")
    title.setToolTip(tr("A linked value follows others (its “=” button); a constraint keeps a fitted value on one side "
                        "of an expression. Both are kept in the model, the saved model and the fit's record."))
    layout.addWidget(title)
    model = editor.model
    contact = QCheckBox(tr("Particles do not overlap (D ≥ 2R, R_HS ≥ R)"), card)
    contact.setObjectName("fitContact")
    contact.setToolTip(tr(CONTACT_TIP))
    contact.setChecked(model.contact)
    contact.toggled.connect(lambda on: editor.set_contact(bool(on)))
    layout.addWidget(contact)
    for path, parameter in model.parameters():
        if parameter.tie:
            layout.addLayout(_line(card, f"{parameter_name(path)} = {parameter.tie}", tr("Unlink"),
                                   lambda _c=False, p=path: editor.link(p, None), f"fitUnlink_{parameter_name(path)}"))
    for index, inequality in enumerate(model.inequalities):
        layout.addLayout(_line(card, inequality.text, tr("Remove this constraint"),
                               lambda _c=False, i=index: editor.remove_constraint(i), f"fitRemoveConstraint_{index}"))
    add = QPushButton(tr("Add Constraint…"), card)
    add.setObjectName("fitAddConstraint")
    add.setToolTip(tr("A fitted value ≥ or ≤ an expression of others, e.g. D_2 >= 2*R_2 + 3"))
    add.clicked.connect(lambda: ask_constraint(editor, add))
    row = QHBoxLayout()
    row.addWidget(add)
    row.addStretch(1)
    layout.addLayout(row)
    return card


def _line(card, text: str, tip: str, slot, name: str) -> QHBoxLayout:
    row = QHBoxLayout()
    label = QLabel(text, card)
    label.setProperty("gimapRole", "muted")
    remove = QToolButton(card)
    remove.setObjectName(name)
    remove.setText("×")
    remove.setAutoRaise(True)
    remove.setToolTip(tip)
    remove.clicked.connect(slot)
    row.addWidget(label, 1)
    row.addWidget(remove)
    return row


def ask_constraint(editor, parent) -> None:
    names = ", ".join(parameter_name(path) for path, _p in editor.model.parameters())
    text, ok = QInputDialog.getText(parent, tr("Add Constraint"), trf(CONSTRAINT_HELP, names=names))
    if ok and text.strip():
        editor.add_constraint(text.strip())


def add_constraint(model, text: str):
    """``model`` with the constraint ``text`` added (``ValueError`` when it cannot be read or names an unknown value)."""
    return with_inequalities(model, (*model.inequalities, Inequality.parse(text)))


def remove_constraint(model, index: int):
    kept = [inequality for number, inequality in enumerate(model.inequalities) if number != index]
    return with_inequalities(model, kept)


def summary(model) -> str:
    """One line of every link and constraint (the Model step's tooltip)."""
    return "; ".join(describe_links(model))


__all__ = ["add_constraint", "ask_constraint", "links_card", "remove_constraint", "summary"]
