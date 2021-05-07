"""
Library defining the standard colours that should be used in GenX. All colours are based on those defined
in the TANGO desktop project, http://tango.freedesktop.org
"""

__author__='Matts Bjorck'

import wx

HIGHLIGHT=0
NORMAL=1
SHADOW=2

class StandardColours:
    highlight=0
    normal=1
    shadow=2
    butter=((0xfc, 0xe9, 0x4f), (0xed, 0xd4, 0x00), (0xc4, 0xa0, 0x00))
    aluminum=((0xee, 0xee, 0xec), (0xd3, 0xd7, 0xcf), (0xba, 0xbd, 0xb6))
    chameleon=((0x8a, 0xe2, 0x34), (0x73, 0xd2, 0x16), (0x4e, 0x9a, 0x06))
    orange=((0xfc, 0xaf, 0x3e), (0xf5, 0x79, 0x00), (0xce, 0x5c, 0x00))
    chocolate=((0xe9, 0xb9, 0x6e), (0xc1, 0x7d, 0x11), (0x8f, 0x59, 0x02))
    sky_blue=((0x72, 0x9f, 0xcf), (0x34, 0x65, 0xa4), (0x20, 0x4a, 0x87))
    plum=((0xad, 0x7f, 0xa8), (0x75, 0x50, 0x7b), (0x5c, 0x35, 0x66))
    slate=((0x88, 0x8a, 0x85), (0x55, 0x57, 0x53), (0x2e, 0x34, 0x36))
    scarlet_red=((0xef, 0x29, 0x29), (0xcc, 0x00, 0x00), (0xa4, 0x00, 0x00))

    translations={'red': 'scarlet_red', 'blue': 'sky_blue', 'green': 'chameleon', 'yellow': 'butter'}

    def get_rgb(self, name, appearance=NORMAL):
        """ Return a colour from the TANGO palette.

        :param name: The name of the colour
        :param appearance: One of the values HIGHLIGHT, NORMAL or SHADOW, default is NORMAL
        :return: A tuple representing the RGB colour
        """

        return getattr(self, self._translate_name(name))[appearance]

    def get_colour(self, name, appearance=NORMAL):
        """ Return the wx.Colour object for named colour.

        :param name:  The name of the colour
        :param appearance: One of the values HIGHLIGHT, NORMAL or SHADOW, default is NORMAL
        :return: wx.Colour object
        """
        rgb=self.get_rgb(name, appearance)
        return wx.Colour(rgb[0], rgb[1], rgb[2])

    def _translate_name(self, name):
        if name.lower() in self.translations:
            return self.translations[name.lower()]
        else:
            return name.lower()

colours=StandardColours()
