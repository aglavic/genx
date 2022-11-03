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
import wx

from wx.svg import SVGimage
from dataclasses import dataclass
from numpy import log, exp, array, sqrt
from genx.models.lib.refl import InstrumentBase, LayerBase, SampleBase, StackBase
from .. import add_on_framework as framework

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

    @property
    def side(self):
        # return lighter color for side of 3d block
        return LColor(min(1.0, self.r+0.35), min(1.0, self.g+0.35), min(1.0, self.b+0.35))

    @property
    def top(self):
        # return lighter color for side of 3d block
        return LColor(min(1.0, self.r+0.2), min(1.0, self.g+0.2), min(1.0, self.b+0.2))

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
            if len(thicknesses[-1])>0:
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
            elif self.show_all or ri==2:
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
            elif self.show_all or ri==2:
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
    use3d: bool

    view_box = array([100, 200])
    move_3d = 8.0
    vanishing_point=(175., -15.)

    def __init__(self, sample: SampleBase, rescale=True, show_all=False, show_one=False, use3d=True):
        self.use3d=use3d
        self.block_generator = BlockGenerator(sample.Stacks, rescale=rescale, show_all=show_all, show_one=show_one)
        self.create_svg()

    def to_vp(self, point0):
        """
        Calculate the x,y-values for a shift from a starting point (x0,y0)
        towards the vanishing point by an x-offset of self.move_3d.
        """
        px, py=self.view_box*0.01
        x0, y0 = point0
        dx = self.move_3d
        vx, vy = self.vanishing_point
        relx = dx/(vx-x0) # releative movement towrads vanishing point
        dy = (vy-y0)*relx
        return ((x0+dx)*px, (y0+dy)*py)

    def add_rect(self, pos:float, si:float, color:LColor=LColor(0., 0., 0.), in_stack=False,
                 width=90):
        px, py=self.view_box*0.01
        if in_stack:
            x= 10
            w = width-5
        else:
            x = 5
            w = width
        if self.use3d:
            w -= 10
            points = [((x+w)*px, pos*py),
                      ((x+w)*px, (pos+si)*py),
                      self.to_vp(((x+w), (pos+si))),
                      self.to_vp(((x+w), pos))]
            side = self.svg.polygon(points=points,
                                    stroke='black', stroke_width=0.5,
                                    fill=str(color.side))
            self.svg.add(side)
        rect = self.svg.rect(
            (x*px, pos*py),
            (w*px, si*py),
            stroke='black', stroke_width=0.5,
            fill=str(color))
        self.svg.add(rect)

    def create_svg(self):
        px, py = self.view_box*0.01
        bg = self.block_generator
        self.svg = svgwrite.Drawing('genx_model.svg', size=("4cm", "8cm"),
                                    viewBox=f'0 0 {self.view_box[0]} {self.view_box[1]}')
        vscale = bg.dtotal / 85. # 90% of image height to be used for layers

        blocks = bg.get_blocks()

        if self.use3d:
            tc = 45
        else:
            tc = 50

        pos = 90.0
        self.add_rect(pos, 100, LColor(0, 0, 0))
        bc = LColor(0,0,0)
        for bi in blocks:
            if isinstance(bi, list):
                block_length = sum([bij.scale/vscale for bij in bi])
                self.add_rect(pos-block_length, block_length, LColor(0.75, 0.75, 0.75), width=15)
                for bij in bi:
                    dij = bij.scale/vscale
                    pos -= dij
                    if bij.thickness==-1:
                        if self.use3d:
                            # add surface polygon
                            shifted = self.to_vp((85, pos+dij))
                            points = [(10*px, (pos+dij)*py),
                                      (85*px, (pos+dij)*py),
                                      shifted,
                                      (shifted[0]-75*px, shifted[1])]
                            side = self.svg.polygon(points=points,
                                                    stroke='black', stroke_width=0.5,
                                                    fill=str(bc.top))
                            self.svg.add(side)
                        paragraph = self.svg.add(self.svg.g(font_size=14))
                        paragraph.add(self.svg.text("...", (tc*px, (pos+dij/3.)*py),
                                                    text_anchor='middle', dominant_baseline='middle'))
                    else:
                        bc = bij.color
                        self.add_rect(pos, dij, bc, in_stack=True)
                        paragraph = self.svg.add(self.svg.g(font_size=14))
                        paragraph.add(self.svg.text(f"{bij.thickness:.0f}", (tc*px, (pos+dij/2.)*py),
                                                    text_anchor='middle', dominant_baseline='middle'))
            else:
                di = bi.scale/vscale
                pos -= di
                bc = bi.color
                self.add_rect(pos, di, bc)
                paragraph = self.svg.add(self.svg.g(font_size=14))
                paragraph.add(self.svg.text(f"{bi.thickness:.0f}", (tc*px, (pos+di/2.)*py),
                                            text_anchor='middle', dominant_baseline='middle'))
        if self.use3d:
            # add surface polygon
            shifted = self.to_vp((85, pos))
            points = [(5*px, pos*py),
                      (85*px, pos*py),
                      shifted,
                      (shifted[0]-80*px, shifted[1])]
            side = self.svg.polygon(points=points,
                                    stroke='black', stroke_width=0.5,
                                    fill=str(bc.top))
            self.svg.add(side)

class SVGPanel(wx.Panel):
    svg_img: SVGimage = None
    last_scale = 1.0

    def __init__(self, parent):
        super().__init__(parent=parent)
        self.Bind(wx.EVT_PAINT, self.OnPaint)

    def OnPaint(self, event: wx.PaintEvent):
        if self.svg_img is not None:
            img = self.svg_img

            dc = wx.PaintDC(self)
            dc.SetBackground(wx.Brush('white'))
            dc.Clear()

            scale = min(self.Size.width/img.width, self.Size.height/img.height)
            if self.last_scale!=scale:
                self.Refresh()
                self.Update()
                self.last_scale=scale
                return

            ctx = wx.GraphicsContext.Create(dc)
            img.RenderToGC(ctx, scale)
        else:
            event.Skip()


class Plugin(framework.Template):
    svg: str = None

    def __init__(self, parent):
        framework.Template.__init__(self, parent)
        self.parent=parent

        # Create the SLD plot
        LG_panel=self.NewPlotFolder('Layer Graphics')
        SA_sizer=wx.BoxSizer(wx.VERTICAL)
        LG_panel.SetSizer(SA_sizer)

        self.img = SVGPanel(LG_panel)
        SA_sizer.Add(self.img, 1, wx.EXPAND | wx.GROW | wx.ALL)

        self.rescale = wx.CheckBox(LG_panel, label='Rescale thickness display')
        self.rescale.SetValue(True)
        SA_sizer.Add(self.rescale, 0, wx.FIXED_MINSIZE)

        self.show_all = wx.RadioButton(LG_panel, label='All layers')
        SA_sizer.Add(self.show_all, 0, wx.FIXED_MINSIZE)

        show_topbot = wx.RadioButton(LG_panel, label='Top/Bottom layers')
        show_topbot.SetValue(True)
        SA_sizer.Add(show_topbot, 0, wx.FIXED_MINSIZE)

        self.show_one = wx.RadioButton(LG_panel, label='Single Repetition')
        SA_sizer.Add(self.show_one, 0, wx.FIXED_MINSIZE)

        self.use3d = wx.CheckBox(LG_panel, label='Pseudo 3d')
        self.use3d.SetValue(True)
        SA_sizer.Add(self.use3d, 0, wx.FIXED_MINSIZE)

        LG_panel.Layout()
        self.OnSimulate(None)

        self.rescale.Bind(wx.EVT_CHECKBOX, self.OnSimulate)
        self.show_all.Bind(wx.EVT_RADIOBUTTON, self.OnSimulate)
        show_topbot.Bind(wx.EVT_RADIOBUTTON, self.OnSimulate)
        self.show_one.Bind(wx.EVT_RADIOBUTTON, self.OnSimulate)
        self.use3d.Bind(wx.EVT_CHECKBOX, self.OnSimulate)

    def OnSimulate(self, event):
        # Calculate and update the sld plot
        model = self.GetModel()
        if model.script_module is None:
            return
        gen=SVGenerator(model.script_module.sample,
                        rescale=self.rescale.GetValue(),
                        show_all=self.show_all.GetValue(),
                        show_one=self.show_one.GetValue(),
                        use3d=self.use3d.GetValue())
        self.svg = gen.svg.tostring()
        self.img.svg_img = SVGimage.CreateFromBytes(self.svg.encode('utf-8'))
        self.img.Refresh()


# from genx.plugins.add_ons import LayerGraphics
# open(r'C:\Users\glavic_a\Downloads\test.svg', 'w').write(LayerGraphics.SVGenerator(model.script_module.sample).svg.tostring())
