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
from numpy import log, array, sqrt, pi
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

    def get_wx(self):
        return wx.Colour(int(self.r*225), int(self.g*255), int(self.b*255))

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
        self.stacks = [si for si in stacks if len(si.Layers)>0]
        self.rescale = rescale
        self.show_all = show_all
        self.show_one = show_one
        self.get_sample_range()

    def rescale_thickness(self, di):
        ldi = log(di)
        ldmin = log(self.dmin)
        ldmax = log(self.dmax)
        if ldmin==ldmax:
            # don't resace if all thicknesses are the same
            return di
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
        color_slds = []

        for stack in self.stacks:
            repetitions.append(int(stack.Repetitions))
            thicknesses.append([li.d for li in stack.Layers])
            color_slds.append([abs(li.dens*li.b)+abs(li.dens*li.f) for li in stack.Layers])
            if len(thicknesses[-1])>0:
                dmin = min(dmin, min(thicknesses[-1]))
                dmax = max(dmax, max(thicknesses[-1]))

        unique_ids={}
        color_ids=[]
        cid = 0
        for group_slds in color_slds:
            group_ids=[]
            for csld in group_slds:
                csld = round(csld, 4)
                if csld not in unique_ids:
                    unique_ids[csld]=cid
                    cid+=1
                group_ids.append(unique_ids[csld])
            color_ids.append(group_ids)

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
                dtotal += dsequence + dsequence / 2
            elif self.show_all or ri==2:
                dtotal += ri*dsequence
            else:
                dtotal += 2*dsequence + dsequence / 2
        self.dtotal = dtotal

    def get_blocks(self):
        """
        Returns a list of blocks with either a thickness or list of thicknesses
        """
        output = []
        for ri, ti, si, cidi in zip(self.repetitions, self.thicknesses, self.scaled, self.color_ids):
            infos = [LInfo(ti_j, si_j, COLORS[cid_j%len(COLORS)]) for ti_j, si_j, cid_j in zip(ti, si, cidi)]
            dsequence = sum(si)
            if ri<2:
                output += infos
            elif self.show_one:
                output.append(infos+[LInfo(-ri, dsequence/2, LColor(1., 1., 1.))])
            elif self.show_all or ri==2:
                output.append(ri*infos)
            else:
                output.append(infos+[LInfo(-ri, dsequence/2, LColor(1., 1., 1.))]+infos)
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
                    if bij.thickness<0:
                        bstr=bstr[:-1]
                        bstr+=' (x%i) '%(-bij.thickness)
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

    unit_precision: int # number of digits behind decimal point for thickness label
    unit_nm: bool # use nm instead of angstrom as unit

    view_box = array([100, 200])
    move_3d = 8.0
    vanishing_point=(175., -15.)
    fontsize = 10

    def __init__(self, sample: SampleBase, rescale=True, show_all=False, show_one=False, use3d=True,
                 unit_precision=1, unit_nm=True, fontsize=10):
        self.use3d = use3d
        self.fontsize = fontsize
        self.unit_precision = unit_precision
        self.unit_nm = unit_nm
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

    def tlabel(self, thickness):
        if self.unit_nm:
            return f"{thickness/10:.{self.unit_precision}f} nm"
        else:
            return f"{thickness:.{self.unit_precision}f} Å"

    def add_rect(self, pos:float, si:float, color:LColor=LColor(0., 0., 0.), in_stack=False,
                 width=90):
        px, py=self.view_box*0.01
        if in_stack:
            x= 5 + self.fontsize
            w = width-self.fontsize
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
                self.add_rect(pos-block_length, block_length, LColor(0.75, 0.75, 0.75), width=10+self.fontsize)
                for bij in bi:
                    dij = bij.scale/vscale
                    pos -= dij
                    if bij.thickness<0:
                        if self.use3d:
                            # add surface polygon
                            shifted = self.to_vp((85, pos+dij))
                            points = [((5+self.fontsize)*px, (pos+dij)*py),
                                      (85*px, (pos+dij)*py),
                                      shifted,
                                      (shifted[0]-(85-5-self.fontsize)*px, shifted[1])]
                            side = self.svg.polygon(points=points,
                                                    stroke='black', stroke_width=0.5,
                                                    fill=str(bc.top))
                            self.svg.add(side)
                        paragraph = self.svg.add(self.svg.g(font_size=self.fontsize))
                        paragraph.add(self.svg.text(" ... ", (tc*px, (pos+dij/3.)*py),
                                                    text_anchor='middle', dominant_baseline='middle'))
                        paragraph = self.svg.add(self.svg.g(font_size=self.fontsize,
                                                            transform=f"translate(5, {(pos+dij)*py}) rotate(-90)"))
                        paragraph.add(self.svg.text(f"(x%i)"%(-bij.thickness), (0, 0),
                                                    text_anchor='middle', dominant_baseline='hanging'))
                    else:
                        bc = bij.color
                        self.add_rect(pos, dij, bc, in_stack=True)
                        paragraph = self.svg.add(self.svg.g(font_size=self.fontsize))
                        paragraph.add(self.svg.text(self.tlabel(bij.thickness), (tc*px, (pos+dij/2.)*py),
                                                    text_anchor='middle', dominant_baseline='middle'))
            else:
                di = bi.scale/vscale
                pos -= di
                bc = bi.color
                self.add_rect(pos, di, bc)
                paragraph = self.svg.add(self.svg.g(font_size=self.fontsize))
                paragraph.add(self.svg.text(self.tlabel(bi.thickness), (tc*px, (pos+di/2.)*py),
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

    def add_rect_gc(self, gc:wx.GraphicsContext, pos:float, si:float, color:LColor=LColor(0., 0., 0.), in_stack=False,
                 width=90):
        px, py=self.view_box*0.01
        gx, gy = gc.GetSize()
        gc_scale = min(gx/self.view_box[0], gy/self.view_box[1])

        if in_stack:
            x= 5 + self.fontsize
            w = width-self.fontsize
        else:
            x = 5
            w = width
        if self.use3d:
            w -= 10
            points = [((x+w)*px, pos*py),
                      ((x+w)*px, (pos+si)*py),
                      self.to_vp(((x+w), (pos+si))),
                      self.to_vp(((x+w), pos))]
            self.draw_polygon_gc(gc, points, color.side)
        gc.SetPen(wx.BLACK_PEN)
        gc.SetBrush(wx.Brush(color.get_wx(), style=wx.BRUSHSTYLE_SOLID))
        gc.DrawRectangle(x*px*gc_scale, pos*py*gc_scale, w*px*gc_scale, si*py*gc_scale)

    def draw_polygon_gc(self, gc:wx.GraphicsContext, points, fill=LColor(1, 1, 1)):
        gx, gy = gc.GetSize()
        gc_scale = min(gx/self.view_box[0], gy/self.view_box[1])

        gc.SetPen(wx.BLACK_PEN)
        gc.SetBrush(wx.Brush(fill.get_wx(), style=wx.BRUSHSTYLE_SOLID))

        points2d=[wx.Point2D(gc_scale*x, gc_scale*y) for x,y in points]
        points2d.append(wx.Point2D(gc_scale*points[0][0], gc_scale*points[0][1]))

        gc.DrawLines(points2d)


    def draw_centered_text(self, gc:wx.GraphicsContext, txt:str, x, y, baseline=False):
        w, h = gc.GetFullTextExtent(txt)[:2]
        if baseline:
            gc.DrawText(txt, int(x-w/2.), int(y))
        else:
            gc.DrawText(txt, int(x-w/2.), int(y-h/2.))

    def render_to_gc(self, gc:wx.GraphicsContext):
        px, py = self.view_box*0.01
        bg = self.block_generator
        gx, gy = gc.GetSize()
        gc_scale = min(gx/self.view_box[0], gy/self.view_box[1])

        vscale = bg.dtotal / 85. # 90% of image height to be used for layers

        blocks = bg.get_blocks()

        gc.SetFont(wx.Font(int(self.fontsize*gc_scale*0.75),
                           wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL), wx.BLACK)

        if self.use3d:
            tc = 45
        else:
            tc = 50

        pos = 90.0
        self.add_rect_gc(gc, pos, 100, LColor(0, 0, 0))
        bc = LColor(0,0,0)
        for bi in blocks:
            if isinstance(bi, list):
                block_length = sum([bij.scale/vscale for bij in bi])
                self.add_rect_gc(gc, pos-block_length, block_length, LColor(0.75, 0.75, 0.75), width=10+self.fontsize)
                for bij in bi:
                    dij = bij.scale/vscale
                    pos -= dij
                    if bij.thickness<0:
                        if self.use3d:
                            # add surface polygon
                            shifted = self.to_vp((85, pos+dij))
                            points = [((5+self.fontsize)*px, (pos+dij)*py),
                                      (85*px, (pos+dij)*py),
                                      shifted,
                                      (shifted[0]-(85-5-self.fontsize)*px, shifted[1])]
                            self.draw_polygon_gc(gc, points, bc)
                        self.draw_centered_text(gc, "...", gc_scale*tc*px, gc_scale*(pos+dij/3.)*py)
                        gc.DrawText(f"(x%i)"%(-bij.thickness), gc_scale*5, gc_scale*(pos+dij)*py, angle=pi/2)
                    else:
                        bc = bij.color
                        self.add_rect_gc(gc, pos, dij, bc, in_stack=True)
                        self.draw_centered_text(gc, self.tlabel(bij.thickness),
                                                gc_scale*tc*px, gc_scale*(pos+dij/2.)*py)
            else:
                di = bi.scale/vscale
                pos -= di
                bc = bi.color
                self.add_rect_gc(gc, pos, di, bc)
                self.draw_centered_text(gc, self.tlabel(bi.thickness),
                                        gc_scale*tc*px, gc_scale*(pos+di/2.)*py)
        if self.use3d:
            # add surface polygon
            shifted = self.to_vp((85, pos))
            points = [(5*px, pos*py),
                      (85*px, pos*py),
                      shifted,
                      (shifted[0]-80*px, shifted[1])]
            self.draw_polygon_gc(gc, points, bc)


class SVGPanel(wx.Panel):
    #svg_img: SVGimage = None
    svg_img: SVGenerator = None
    last_scale = 1.0

    def __init__(self, parent):
        super().__init__(parent=parent, style=wx.FULL_REPAINT_ON_RESIZE)
        self.Bind(wx.EVT_PAINT, self.OnPaint)

    def OnPaint(self, event: wx.PaintEvent):
        if self.svg_img is not None:
            img = self.svg_img

            dc = wx.PaintDC(self)
            dc.SetBackground(wx.Brush('white'))
            dc.Clear()

            ctx = wx.GraphicsContext.Create(dc)
            img.render_to_gc(ctx)
            #img.RenderToGC(ctx, scale)
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

        bot_sizer=wx.BoxSizer(wx.HORIZONTAL)
        SA_sizer.Add(bot_sizer, 0, wx.FIXED_MINSIZE)
        left_sizer=wx.BoxSizer(wx.VERTICAL)
        bot_sizer.Add(left_sizer, 1, wx.EXPAND | wx.GROW | wx.FIXED_MINSIZE)
        bot_sizer.AddSpacer(4)
        mid_sizer=wx.BoxSizer(wx.VERTICAL)
        bot_sizer.Add(mid_sizer, 1, wx.EXPAND | wx.GROW | wx.FIXED_MINSIZE)
        bot_sizer.AddSpacer(4)
        right_sizer=wx.BoxSizer(wx.VERTICAL)
        bot_sizer.Add(right_sizer, 1, wx.EXPAND | wx.GROW | wx.FIXED_MINSIZE)


        self.rescale = wx.CheckBox(LG_panel, label='Rescale thickness display')
        self.rescale.SetValue(True)
        left_sizer.Add(self.rescale, 0, wx.FIXED_MINSIZE)

        self.show_all = wx.RadioButton(LG_panel, label='All layers')
        left_sizer.Add(self.show_all, 0, wx.FIXED_MINSIZE)

        show_topbot = wx.RadioButton(LG_panel, label='Top/Bottom layers')
        show_topbot.SetValue(True)
        left_sizer.Add(show_topbot, 0, wx.FIXED_MINSIZE)

        self.show_one = wx.RadioButton(LG_panel, label='Single Repetition')
        left_sizer.Add(self.show_one, 0, wx.FIXED_MINSIZE)

        self.use3d = wx.CheckBox(LG_panel, label='Pseudo 3d')
        self.use3d.SetValue(True)
        mid_sizer.Add(self.use3d, 0, wx.FIXED_MINSIZE)
        self.unit_nm = wx.CheckBox(LG_panel, label='nm-unit')
        self.unit_nm.SetValue(True)
        mid_sizer.Add(self.unit_nm, 0, wx.FIXED_MINSIZE)
        self.unit_precision = wx.SpinCtrl(LG_panel, value="1", min=0, max=5)
        mid_sizer.Add(self.unit_precision, 0, wx.FIXED_MINSIZE)

        self.fontsize = wx.SpinCtrl(LG_panel, value="10", min=1, max=20)
        right_sizer.Add(self.fontsize, 0, wx.FIXED_MINSIZE)
        export_button = wx.Button(LG_panel, label='Save to SVG...')
        right_sizer.Add(export_button)
        copy_button = wx.Button(LG_panel, label='Copy to Clipboard')
        right_sizer.Add(copy_button)

        LG_panel.Layout()
        try:
            self.OnSimulate(None)
        except Exception:
            pass

        self.rescale.Bind(wx.EVT_CHECKBOX, self.OnSimulate)
        self.show_all.Bind(wx.EVT_RADIOBUTTON, self.OnSimulate)
        show_topbot.Bind(wx.EVT_RADIOBUTTON, self.OnSimulate)
        self.show_one.Bind(wx.EVT_RADIOBUTTON, self.OnSimulate)
        self.use3d.Bind(wx.EVT_CHECKBOX, self.OnSimulate)
        self.unit_nm.Bind(wx.EVT_CHECKBOX, self.OnSimulate)
        self.unit_precision.Bind(wx.EVT_SPINCTRL, self.OnSimulate)
        self.fontsize.Bind(wx.EVT_SPINCTRL, self.OnSimulate)
        export_button.Bind(wx.EVT_BUTTON, self.ExportSVG)
        copy_button.Bind(wx.EVT_BUTTON, self.CopyImage)

    def OnSimulate(self, event):
        # Calculate and update the sld plot
        model = self.GetModel()
        if model.script_module is None:
            return
        gen=SVGenerator(model.script_module.sample,
                        rescale=self.rescale.GetValue(),
                        show_all=self.show_all.GetValue(),
                        show_one=self.show_one.GetValue(),
                        use3d=self.use3d.GetValue(),
                        unit_nm=self.unit_nm.GetValue(),
                        unit_precision=int(self.unit_precision.GetValue()),
                        fontsize=self.fontsize.GetValue(),
                        )
        self.svg = gen.svg.tostring()
        #self.img.svg_img = SVGimage.CreateFromBytes(self.svg.encode('utf-8'))
        self.img.svg_img = gen
        self.img.Refresh()

    def ExportSVG(self, event):
        dlg = wx.FileDialog(self.img, message="Save layer sketch image", defaultFile="",
                            wildcard="SVG Image|*.svg",
                            style=wx.FD_SAVE | wx.FD_CHANGE_DIR | wx.FD_OVERWRITE_PROMPT
                            )
        if dlg.ShowModal()==wx.ID_OK:
            path = dlg.GetPath()
            with open(path, 'w', encoding='utf-8') as fh:
                fh.write(self.svg)
        dlg.Destroy()

    def CopyImage(self, event):
        bmp = wx.Bitmap(wx.Size(300, 600), depth=32)
        if hasattr(bmp, 'SetScaleFactor'):
            bmp.SetScaleFactor(2.0)

        memdc = wx.MemoryDC(bmp)
        memdc.SetBackground(wx.Brush(wx.Colour(255, 255, 255, 0)))
        memdc.Clear()

        ctx = wx.GraphicsContext.Create(memdc)

        self.img.svg_img.render_to_gc(ctx)
        memdc.SelectObject(wx.NullBitmap)

        bmp_obj = wx.BitmapDataObject()
        bmp_obj.SetBitmap(bmp)

        if not wx.TheClipboard.IsOpened():
            open_success = wx.TheClipboard.Open()
            if open_success:
                wx.TheClipboard.SetData(bmp_obj)
                wx.TheClipboard.Close()
                wx.TheClipboard.Flush()
