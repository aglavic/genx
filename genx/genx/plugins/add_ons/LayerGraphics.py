# -*- coding: utf8 -*-
'''
=============
LayerGraphics
=============

A plugin to create sketch graphics to vizualize the layer structure.
The result is an SVG image with automatically generated colors and sizes.

Thicknesses can either be linear or mapped through a scaling funciton to
avoid large variation if thicknesses are too different.
'''

import svgwrite

from dataclasses import dataclass
from numpy import log, exp, array, sqrt
from genx.models.lib.refl import InstrumentBase, LayerBase, SampleBase, StackBase

try:
    # noinspection PyUnresolvedReferences
    from typing import List
except ImportError:
    List = object

@dataclass
class LColor:
    r: float
    g: float
    b: float

    def __add__(self, other):
        out = LColor(self.r+other.r, self.g+other.g, self.b+other.b)

    def normalize(self):
        norm = sqrt((array([self.r, self.g, self.b])**2).sum())
        self.r/=norm
        self.g/=norm
        self.b/=norm

    def __str__(self):
        return f"rgb({self.r*255:.0f}, {self.g*255:.0f}, {self.b*255:.0f})"

@dataclass
class LInfo:
    thickness: float
    scale: float
    color: LColor

COLORS = [LColor(1., 0., 0.), LColor(0., 1., 0.), LColor(0., 0., 1.),
          LColor(1., 1., 0.), LColor(1., 0., 1.), LColor(0., 1., 1.)]

class BlockGenerator:
    stacks: List[StackBase]
    rescale: bool  # apply the mapping function to reduce visible variation of size
    show_all: bool # show all layers in a repeating stack separately
    show_one: bool # show just one sequence in a repreating stack when above 1 repetition

    dmin: float
    dmax: float
    dtotal: float

    repetitions: List[int]
    thicknesses: List[List[float]]
    scaled: List[List[float]]

    min_rescale = 30.
    max_rescale = 100.

    def __init__(self, stacks: List[StackBase], rescale=True, show_all=False, show_one=False):
        self.stacks = stacks
        self.rescale = rescale
        self.show_all = show_all
        self.show_one = show_one
        self.get_sample_range()

    def rescale_thickness(self, di):
        ldi = log(di)
        ldmin = log(self.dmin)
        ldmax = log(self.dmax)
        ld = (ldi-ldmin)/(ldmax-ldmin)
        sd = self.min_rescale + (self.max_rescale-self.min_rescale)*ld
        return sd

    def get_sample_range(self):
        # return total thickness, smallest and largest layer
        dmin = 1.0e5
        dmax = 0.
        dtotal = 0.

        repetitions = []
        thicknesses = []
        color_ids = []
        cid = 0

        for stack in self.stacks:
            repetitions.append(int(stack.Repetitions))
            thicknesses.append([li.d for li in stack.Layers])
            color_ids.append([cid+i for i, li in enumerate(stack.Layers)])
            cid+=len(color_ids[-1])
            dmin = min(dmin, min(thicknesses[-1]))
            dmax = max(dmax, max(thicknesses[-1]))
        self.dmin=dmin
        self.dmax=dmax
        self.repetitions = repetitions
        self.color_ids = color_ids

        if self.rescale:
            sthicknesses = [list(map(self.rescale_thickness, ti)) for ti in thicknesses]
        else:
            sthicknesses = thicknesses
        self.thicknesses = thicknesses
        self.scaled = sthicknesses

        for ri, ti in zip(repetitions, sthicknesses):
            dsequence=sum(ti)
            if ri<2:
                dtotal+=dsequence
            elif self.show_one:
                dtotal += dsequence
            elif self.show_all:
                dtotal += ri*dsequence
            else:
                dtotal += 2*dsequence + self.min_rescale
        self.dtotal = dtotal

    def get_blocks(self):
        """
        Returns a list of blocks with either a thickness or list of thicknesses
        """
        output = []
        for ri, ti, si, cidi in zip(self.repetitions, self.thicknesses, self.scaled, self.color_ids):
            infos = [LInfo(ti_j, si_j, COLORS[cid_j%len(COLORS)]) for ti_j, si_j, cid_j in zip(ti, si, cidi)]
            if ri<2:
                output += infos
            elif self.show_one:
                output.append(infos)
            elif self.show_all:
                output.append(ri*infos)
            else:
                output.append(infos+[LInfo(-1., self.min_rescale, LColor(1., 1., 1.))]+infos)
        return output

    def __repr__(self):
        blocks = self.get_blocks()
        bstr = 'substrate |'
        sstr = ' |'
        for bi in blocks:
            if isinstance(bi, list):
                bstr+=f'|'
                sstr+=f'|'
                for bij in bi:
                    sstr+=f' {bij.scale:.0f} |'
                    if bij.thickness==-1:
                        bstr=bstr[:-1]
                        bstr+=' ... '
                    else:
                        bstr+=f' {bij.thickness:.0f} |'
                bstr += f'|'
                sstr += f'|'
            else:
                bstr+=f' {bi.thickness:.0f} |'
                sstr += f' {bi.scale:.0f} |'
        bstr += ' ambient'
        return f'BlockGenerator(\n               {bstr}\n               {sstr})'

class SVGenerator:
    block_generator: BlockGenerator
    svg: svgwrite.Drawing

    def __init__(self, sample: SampleBase, rescale=True, show_all=False, show_one=False):
        self.block_generator = BlockGenerator(sample.Stacks, rescale=rescale, show_all=show_all, show_one=show_one)
        self.create_svg()

    def get_rect(self, pos:float, di:float, si:float, color:LColor=LColor(0., 0., 0.), in_stack=False):
        if in_stack:
            x= 10
            w = 80
        else:
            x = 5
            w = 90
        return self.svg.rect(
            (x*svgwrite.percent, pos*svgwrite.percent),
            (w*svgwrite.percent, si*svgwrite.percent),
            stroke='darkgray', stroke_width=3,
            fill=str(color))

    def create_svg(self):
        bg = self.block_generator
        self.svg = svgwrite.Drawing('test.svg', size=("400px", "800px"))
        vscale = bg.dtotal / 90.

        blocks = bg.get_blocks()

        pos = 95.0
        for bi in blocks:
            if isinstance(bi, list):
                block_start = pos
                for bij in bi:
                    dij = bij.scale/vscale
                    pos -= dij
                    if bij.thickness==-1:
                        paragraph = self.svg.add(self.svg.g(font_size=20))
                        paragraph.add(self.svg.text("...", (50*svgwrite.percent, (pos+dij/2.)*svgwrite.percent),
                                                    text_anchor='middle', dominant_baseline='middle'))
                    else:
                        self.svg.add(self.get_rect(pos, bij.thickness, dij, bij.color, in_stack=True))
                        paragraph = self.svg.add(self.svg.g(font_size=20))
                        paragraph.add(self.svg.text(f"{bij.thickness:.0f}",
                                                    (50*svgwrite.percent, (pos+dij/2.)*svgwrite.percent),
                                                    text_anchor='middle', dominant_baseline='middle'))
            else:
                di = bi.scale/vscale
                pos -= di
                self.svg.add(self.get_rect(pos, bi.thickness, di, bi.color))
                paragraph = self.svg.add(self.svg.g(font_size=20))
                paragraph.add(self.svg.text(f"{bi.thickness:.0f}",
                                            (50*svgwrite.percent, (pos+di/2.)*svgwrite.percent),
                                            text_anchor='middle', dominant_baseline='middle'))


# from genx.plugins.add_ons import LayerGraphics
# open(r'C:\Users\glavic_a\Downloads\test.svg', 'w').write(LayerGraphics.SVGenerator(model.script_module.sample).svg.tostring())
