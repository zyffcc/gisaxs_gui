"""The 230 space-group types as (number, Hermann–Mauguin symbol, Hall symbol), in the settings of *International
Tables* A, with the common alternative settings.

Read by ``space_groups.py``, which generates the operations from the Hall symbol (``hall.py``). Qualifiers after
a colon: ``:1`` and ``:2`` the two origin choices of the groups that have two (choice 2: the inversion centre at
the origin), ``:H`` and ``:R`` the hexagonal and the rhombohedral axes of an R group. Monoclinic groups are listed
with unique axis b and their full symbol (``P 1 21/c 1``); ``P 21/c`` is the same group. Orthorhombic P groups
in the other five axis settings (``P b n m`` for P n m a) are derived in ``space_groups.py``.

``tests/test_space_groups.py`` checks every entry: the number of general positions against the point group and
lattice of its symbol, the glides and screws of the symbol where they are unambiguous, and the extinctions of the
groups that matter most for thin films."""

from __future__ import annotations

SPACE_GROUPS: tuple[tuple[int, str, str], ...] = (
    (1, "P 1", "P 1"), (2, "P -1", "-P 1"),
    (3, "P 1 2 1", "P 2y"), (4, "P 1 21 1", "P 2yb"), (5, "C 1 2 1", "C 2y"), (5, "I 1 2 1", "I 2y"),
    (6, "P 1 m 1", "P -2y"), (7, "P 1 c 1", "P -2yc"), (7, "P 1 n 1", "P -2yac"), (7, "P 1 a 1", "P -2ya"),
    (8, "C 1 m 1", "C -2y"), (8, "I 1 m 1", "I -2y"), (9, "C 1 c 1", "C -2yc"), (9, "I 1 a 1", "I -2ya"),
    (10, "P 1 2/m 1", "-P 2y"), (11, "P 1 21/m 1", "-P 2yb"), (12, "C 1 2/m 1", "-C 2y"), (12, "I 1 2/m 1", "-I 2y"),
    (13, "P 1 2/c 1", "-P 2yc"), (13, "P 1 2/n 1", "-P 2yac"), (13, "P 1 2/a 1", "-P 2ya"),
    (14, "P 1 21/c 1", "-P 2ybc"), (14, "P 1 21/n 1", "-P 2yn"), (14, "P 1 21/a 1", "-P 2yab"),
    (15, "C 1 2/c 1", "-C 2yc"), (15, "I 1 2/a 1", "-I 2ya"),
    (16, "P 2 2 2", "P 2 2"), (17, "P 2 2 21", "P 2c 2"), (18, "P 21 21 2", "P 2 2ab"),
    (19, "P 21 21 21", "P 2ac 2ab"), (20, "C 2 2 21", "C 2c 2"), (21, "C 2 2 2", "C 2 2"), (22, "F 2 2 2", "F 2 2"),
    (23, "I 2 2 2", "I 2 2"), (24, "I 21 21 21", "I 2b 2c"),
    (25, "P m m 2", "P 2 -2"), (26, "P m c 21", "P 2c -2"), (27, "P c c 2", "P 2 -2c"), (28, "P m a 2", "P 2 -2a"),
    (29, "P c a 21", "P 2c -2ac"), (30, "P n c 2", "P 2 -2bc"), (31, "P m n 21", "P 2ac -2"),
    (32, "P b a 2", "P 2 -2ab"), (33, "P n a 21", "P 2c -2n"), (34, "P n n 2", "P 2 -2n"), (35, "C m m 2", "C 2 -2"),
    (36, "C m c 21", "C 2c -2"), (37, "C c c 2", "C 2 -2c"), (38, "A m m 2", "A 2 -2"), (39, "A e m 2", "A 2 -2b"),
    (40, "A m a 2", "A 2 -2a"), (41, "A e a 2", "A 2 -2ab"), (42, "F m m 2", "F 2 -2"), (43, "F d d 2", "F 2 -2d"),
    (44, "I m m 2", "I 2 -2"), (45, "I b a 2", "I 2 -2c"), (46, "I m a 2", "I 2 -2a"),
    (47, "P m m m", "-P 2 2"), (48, "P n n n :1", "P 2 2 -1n"), (48, "P n n n :2", "-P 2ab 2bc"),
    (49, "P c c m", "-P 2 2c"), (50, "P b a n :1", "P 2 2 -1ab"), (50, "P b a n :2", "-P 2ab 2b"),
    (51, "P m m a", "-P 2a 2a"), (52, "P n n a", "-P 2a 2bc"), (53, "P m n a", "-P 2ac 2"),
    (54, "P c c a", "-P 2a 2ac"), (55, "P b a m", "-P 2 2ab"), (56, "P c c n", "-P 2ab 2ac"),
    (57, "P b c m", "-P 2c 2b"), (58, "P n n m", "-P 2 2n"), (59, "P m m n :1", "P 2 2ab -1ab"),
    (59, "P m m n :2", "-P 2ab 2a"), (60, "P b c n", "-P 2n 2ab"), (61, "P b c a", "-P 2ac 2ab"),
    (62, "P n m a", "-P 2ac 2n"), (63, "C m c m", "-C 2c 2"), (64, "C m c e", "-C 2ac 2"), (65, "C m m m", "-C 2 2"),
    (66, "C c c m", "-C 2 2c"), (67, "C m m e", "-C 2a 2"), (68, "C c c e :1", "C 2 2 -1ac"),
    (68, "C c c e :2", "-C 2a 2ac"), (69, "F m m m", "-F 2 2"), (70, "F d d d :1", "F 2 2 -1d"),
    (70, "F d d d :2", "-F 2uv 2vw"), (71, "I m m m", "-I 2 2"), (72, "I b a m", "-I 2 2c"), (73, "I b c a", "-I 2b 2c"),
    (74, "I m m a", "-I 2b 2"),
    (75, "P 4", "P 4"), (76, "P 41", "P 4w"), (77, "P 42", "P 4c"), (78, "P 43", "P 4cw"), (79, "I 4", "I 4"),
    (80, "I 41", "I 4bw"), (81, "P -4", "P -4"), (82, "I -4", "I -4"), (83, "P 4/m", "-P 4"), (84, "P 42/m", "-P 4c"),
    (85, "P 4/n :1", "P 4ab -1ab"), (85, "P 4/n :2", "-P 4a"), (86, "P 42/n :1", "P 4n -1n"),
    (86, "P 42/n :2", "-P 4bc"), (87, "I 4/m", "-I 4"), (88, "I 41/a :1", "I 4bw -1bw"), (88, "I 41/a :2", "-I 4ad"),
    (89, "P 4 2 2", "P 4 2"), (90, "P 4 21 2", "P 4ab 2ab"), (91, "P 41 2 2", "P 4w 2c"),
    (92, "P 41 21 2", "P 4abw 2nw"), (93, "P 42 2 2", "P 4c 2"), (94, "P 42 21 2", "P 4n 2n"),
    (95, "P 43 2 2", "P 4cw 2c"), (96, "P 43 21 2", "P 4nw 2abw"), (97, "I 4 2 2", "I 4 2"),
    (98, "I 41 2 2", "I 4bw 2bw"), (99, "P 4 m m", "P 4 -2"), (100, "P 4 b m", "P 4 -2ab"),
    (101, "P 42 c m", "P 4c -2c"), (102, "P 42 n m", "P 4n -2n"), (103, "P 4 c c", "P 4 -2c"),
    (104, "P 4 n c", "P 4 -2n"), (105, "P 42 m c", "P 4c -2"), (106, "P 42 b c", "P 4c -2ab"),
    (107, "I 4 m m", "I 4 -2"), (108, "I 4 c m", "I 4 -2c"), (109, "I 41 m d", "I 4bw -2"),
    (110, "I 41 c d", "I 4bw -2c"), (111, "P -4 2 m", "P -4 2"), (112, "P -4 2 c", "P -4 2c"),
    (113, "P -4 21 m", "P -4 2ab"), (114, "P -4 21 c", "P -4 2n"), (115, "P -4 m 2", "P -4 -2"),
    (116, "P -4 c 2", "P -4 -2c"), (117, "P -4 b 2", "P -4 -2ab"), (118, "P -4 n 2", "P -4 -2n"),
    (119, "I -4 m 2", "I -4 -2"), (120, "I -4 c 2", "I -4 -2c"), (121, "I -4 2 m", "I -4 2"),
    (122, "I -4 2 d", "I -4 2bw"), (123, "P 4/m m m", "-P 4 2"), (124, "P 4/m c c", "-P 4 2c"),
    (125, "P 4/n b m :1", "P 4 2 -1ab"), (125, "P 4/n b m :2", "-P 4a 2b"), (126, "P 4/n n c :1", "P 4 2 -1n"),
    (126, "P 4/n n c :2", "-P 4a 2bc"), (127, "P 4/m b m", "-P 4 2ab"), (128, "P 4/m n c", "-P 4 2n"),
    (129, "P 4/n m m :1", "P 4ab 2ab -1ab"), (129, "P 4/n m m :2", "-P 4a 2a"),
    (130, "P 4/n c c :1", "P 4ab 2n -1ab"), (130, "P 4/n c c :2", "-P 4a 2ac"), (131, "P 42/m m c", "-P 4c 2"),
    (132, "P 42/m c m", "-P 4c 2c"), (133, "P 42/n b c :1", "P 4n 2c -1n"), (133, "P 42/n b c :2", "-P 4ac 2b"),
    (134, "P 42/n n m :1", "P 4n 2 -1n"), (134, "P 42/n n m :2", "-P 4ac 2bc"), (135, "P 42/m b c", "-P 4c 2ab"),
    (136, "P 42/m n m", "-P 4n 2n"), (137, "P 42/n m c :1", "P 4n 2n -1n"), (137, "P 42/n m c :2", "-P 4ac 2a"),
    (138, "P 42/n c m :1", "P 4n 2ab -1n"), (138, "P 42/n c m :2", "-P 4ac 2ac"), (139, "I 4/m m m", "-I 4 2"),
    (140, "I 4/m c m", "-I 4 2c"), (141, "I 41/a m d :1", "I 4bw 2bw -1bw"), (141, "I 41/a m d :2", "-I 4bd 2"),
    (142, "I 41/a c d :1", "I 4bw 2aw -1bw"), (142, "I 41/a c d :2", "-I 4bd 2c"),
    (143, "P 3", "P 3"), (144, "P 31", "P 31"), (145, "P 32", "P 32"), (146, "R 3 :H", "R 3"), (146, "R 3 :R", "P 3*"),
    (147, "P -3", "-P 3"), (148, "R -3 :H", "-R 3"), (148, "R -3 :R", "-P 3*"), (149, "P 3 1 2", "P 3 2"),
    (150, "P 3 2 1", 'P 3 2"'), (151, "P 31 1 2", "P 31 2c (0 0 1)"), (152, "P 31 2 1", 'P 31 2"'),
    (153, "P 32 1 2", "P 32 2c (0 0 -1)"), (154, "P 32 2 1", 'P 32 2"'), (155, "R 3 2 :H", 'R 3 2"'),
    (155, "R 3 2 :R", "P 3* 2"), (156, "P 3 m 1", 'P 3 -2"'), (157, "P 3 1 m", "P 3 -2"), (158, "P 3 c 1", 'P 3 -2"c'),
    (159, "P 3 1 c", "P 3 -2c"), (160, "R 3 m :H", 'R 3 -2"'), (160, "R 3 m :R", "P 3* -2"),
    (161, "R 3 c :H", 'R 3 -2"c'), (161, "R 3 c :R", "P 3* -2n"), (162, "P -3 1 m", "-P 3 2"),
    (163, "P -3 1 c", "-P 3 2c"), (164, "P -3 m 1", '-P 3 2"'), (165, "P -3 c 1", '-P 3 2"c'),
    (166, "R -3 m :H", '-R 3 2"'), (166, "R -3 m :R", "-P 3* 2"), (167, "R -3 c :H", '-R 3 2"c'),
    (167, "R -3 c :R", "-P 3* 2n"),
    (168, "P 6", "P 6"), (169, "P 61", "P 61"), (170, "P 65", "P 65"), (171, "P 62", "P 62"), (172, "P 64", "P 64"),
    (173, "P 63", "P 6c"), (174, "P -6", "P -6"), (175, "P 6/m", "-P 6"), (176, "P 63/m", "-P 6c"),
    (177, "P 6 2 2", "P 6 2"), (178, "P 61 2 2", "P 61 2 (0 0 -1)"), (179, "P 65 2 2", "P 65 2 (0 0 1)"),
    (180, "P 62 2 2", "P 62 2c (0 0 1)"), (181, "P 64 2 2", "P 64 2c (0 0 -1)"), (182, "P 63 2 2", "P 6c 2c"),
    (183, "P 6 m m", "P 6 -2"), (184, "P 6 c c", "P 6 -2c"), (185, "P 63 c m", "P 6c -2"),
    (186, "P 63 m c", "P 6c -2c"), (187, "P -6 m 2", "P -6 2"), (188, "P -6 c 2", "P -6c 2"),
    (189, "P -6 2 m", "P -6 -2"), (190, "P -6 2 c", "P -6c -2c"), (191, "P 6/m m m", "-P 6 2"),
    (192, "P 6/m c c", "-P 6 2c"), (193, "P 63/m c m", "-P 6c 2"), (194, "P 63/m m c", "-P 6c 2c"),
    (195, "P 2 3", "P 2 2 3"), (196, "F 2 3", "F 2 2 3"), (197, "I 2 3", "I 2 2 3"), (198, "P 21 3", "P 2ac 2ab 3"),
    (199, "I 21 3", "I 2b 2c 3"), (200, "P m -3", "-P 2 2 3"), (201, "P n -3 :1", "P 2 2 3 -1n"),
    (201, "P n -3 :2", "-P 2ab 2bc 3"), (202, "F m -3", "-F 2 2 3"), (203, "F d -3 :1", "F 2 2 3 -1d"),
    (203, "F d -3 :2", "-F 2uv 2vw 3"), (204, "I m -3", "-I 2 2 3"), (205, "P a -3", "-P 2ac 2ab 3"),
    (206, "I a -3", "-I 2b 2c 3"), (207, "P 4 3 2", "P 4 2 3"), (208, "P 42 3 2", "P 4n 2 3"),
    (209, "F 4 3 2", "F 4 2 3"), (210, "F 41 3 2", "F 4d 2 3"), (211, "I 4 3 2", "I 4 2 3"),
    (212, "P 43 3 2", "P 4acd 2ab 3"), (213, "P 41 3 2", "P 4bd 2ab 3"), (214, "I 41 3 2", "I 4bd 2c 3"),
    (215, "P -4 3 m", "P -4 2 3"), (216, "F -4 3 m", "F -4 2 3"), (217, "I -4 3 m", "I -4 2 3"),
    (218, "P -4 3 n", "P -4n 2 3"), (219, "F -4 3 c", "F -4c 2 3"), (220, "I -4 3 d", "I -4bd 2c 3"),
    (221, "P m -3 m", "-P 4 2 3"), (222, "P n -3 n :1", "P 4 2 3 -1n"), (222, "P n -3 n :2", "-P 4a 2bc 3"),
    (223, "P m -3 n", "-P 4n 2 3"), (224, "P n -3 m :1", "P 4n 2 3 -1n"), (224, "P n -3 m :2", "-P 4bc 2bc 3"),
    (225, "F m -3 m", "-F 4 2 3"), (226, "F m -3 c", "-F 4c 2 3"), (227, "F d -3 m :1", "F 4d 2 3 -1d"),
    (227, "F d -3 m :2", "-F 4vw 2vw 3"), (228, "F d -3 c :1", "F 4d 2 3 -1cd"), (228, "F d -3 c :2", "-F 4cvw 2vw 3"),
    (229, "I m -3 m", "-I 4 2 3"), (230, "I a -3 d", "-I 4bd 2c 3"),
)
"""(International Tables number, Hermann–Mauguin symbol with spaces and qualifier, Hall symbol)."""

OLD_NAMES = {"C m c a": "C m c e", "C m m a": "C m m e", "C c c a :1": "C c c e :1", "C c c a :2": "C c c e :2",
             "A b m 2": "A e m 2", "A b a 2": "A e a 2", "C c c a": "C c c e"}
"""Symbols of International Tables before 2002 (the double glide plane e was then written a or b)."""

__all__ = ["OLD_NAMES", "SPACE_GROUPS"]
