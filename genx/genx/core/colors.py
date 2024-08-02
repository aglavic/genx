"""
Define color specific constants.
"""


class CyclicList(list):
    # A list that can be indexed with larger values than its length, wrapping around
    def __getitem__(self, item):
        return list.__getitem__(self, item % len(self))

    def __eq__(self, other):
        return list.__eq__(self, other)

    def __hash__(self):
        return hash(tuple(self))


COLOR_CYCLES = {
    "none": None,
    "red/blue": CyclicList(
        [
            ((1.0, 0.2, 0.2), (0.5, 0, 0)),  # red
            ((0.3, 0.3, 1.0), (0, 0, 0.5)),  # blue
        ]
    ),
    "red/blue/orange/green": CyclicList(
        [
            ((1.0, 0.2, 0.2), (0.5, 0, 0)),  # red
            ((0.3, 0.3, 1.0), (0, 0, 0.5)),  # blue
            ((1.0, 0.6, 0.2), (0.5, 0.25, 0)),  # orange
            ((0.2, 0.8, 0.2), (0, 0.4, 0)),  # green
        ]
    ),
    "rainbow": CyclicList(
        [
            ((1.0, 0.2, 0.2), (0.5, 0, 0)),  # red
            ((1.0, 0.6, 0.2), (0.5, 0.25, 0)),  # orange
            ((0.9, 0.9, 0.2), (0.5, 0.5, 0)),  # yellow
            ((0.2, 1.0, 0.2), (0, 0.5, 0)),  # green
            ((0.3, 0.3, 1.0), (0, 0, 0.5)),  # blue
            ((0.6, 0.2, 0.8), (0.3, 0, 0.4)),  # purple
        ]
    ),
}
