"""
=============
LayerGraphics
=============

A plugin to create sketch graphics to visualize the layer structure.
The result is an SVG image with automatically generated colors and sizes.

Qt port.
"""

from dataclasses import dataclass

import svgwrite
from numpy import array, log, sqrt
from PySide6 import QtCore, QtGui, QtSvg, QtWidgets

from genx.models.lib.refl_base import SampleBase, StackBase
from genx.plugins import add_on_framework as framework


@dataclass
class LColor:
    r: float
    g: float
    b: float

    def normalize(self):
        norm = sqrt((array([self.r, self.g, self.b]) ** 2).sum())
        self.r /= norm
        self.g /= norm
        self.b /= norm

    @property
    def side(self):
        return LColor(min(1.0, self.r + 0.35), min(1.0, self.g + 0.35), min(1.0, self.b + 0.35))

    @property
    def top(self):
        return LColor(min(1.0, self.r + 0.2), min(1.0, self.g + 0.2), min(1.0, self.b + 0.2))

    def __str__(self):
        return f"rgb({self.r*255:.0f}, {self.g*255:.0f}, {self.b*255:.0f})"


@dataclass
class LInfo:
    thickness: float
    scale: float
    color: LColor
    name: str
    sld: str


COLORS = [
    LColor(1.0, 0.0, 0.0),
    LColor(0.0, 1.0, 0.0),
    LColor(0.0, 0.0, 1.0),
    LColor(1.0, 1.0, 0.0),
    LColor(1.0, 0.0, 1.0),
    LColor(0.0, 1.0, 1.0),
]


class BlockGenerator:
    stacks: list[StackBase]
    rescale: bool
    is_xray: bool
    show_all: bool
    show_one: bool

    dmin: float
    dmax: float
    dtotal: float

    repetitions: list[int]
    thicknesses: list[list[float]]
    scaled: list[list[float]]

    min_rescale = 30.0
    max_rescale = 100.0

    def __init__(self, stacks: list[StackBase], layer_names, is_xray=True, rescale=True, show_all=False, show_one=False):
        self.stacks = [si for si in stacks if len(si.Layers) > 0]
        self.layer_names = layer_names
        self.is_xray = is_xray
        self.rescale = rescale
        self.show_all = show_all
        self.show_one = show_one
        self.get_sample_range()

    def rescale_thickness(self, di):
        ldi = log(di)
        ldmin = log(self.dmin)
        ldmax = log(self.dmax)
        if ldmin == ldmax:
            return di
        ld = (ldi - ldmin) / (ldmax - ldmin)
        sd = self.min_rescale + (self.max_rescale - self.min_rescale) * ld
        return sd

    def get_sample_range(self):
        dmin = 1.0e5
        dmax = 0.0
        dtotal = 0.0

        repetitions = []
        thicknesses = []
        color_slds = []
        names = []
        slds = []

        for stack in self.stacks:
            repetitions.append(int(stack.Repetitions))
            thicknesses.append([li.d for li in stack.Layers])
            names.append([self.layer_names.get(id(li), "").replace("_", " ") for li in stack.Layers])
            if self.is_xray:
                slds.append([f"{li.dens*li.f.real:.2f} rₑ/Å³" for li in stack.Layers])
                color_slds.append([abs(li.dens * li.f) for li in stack.Layers])
            else:
                slds.append([f"{li.dens*li.b.real*10.:.2f} 10⁻⁶ Å⁻²" for li in stack.Layers])
                color_slds.append([abs(li.dens * li.b) for li in stack.Layers])
            if len(thicknesses[-1]) > 0:
                dmin = min(dmin, min(thicknesses[-1]))
                dmax = max(dmax, max(thicknesses[-1]))

        unique_ids = {}
        color_ids = []
        cid = 0
        for group_slds in color_slds:
            group_ids = []
            for csld in group_slds:
                csld = round(csld, 4)
                if csld not in unique_ids:
                    unique_ids[csld] = cid
                    cid += 1
                group_ids.append(unique_ids[csld])
            color_ids.append(group_ids)

        self.dmin = dmin
        self.dmax = dmax
        self.repetitions = repetitions
        self.color_ids = color_ids

        if self.rescale:
            sthicknesses = [list(map(self.rescale_thickness, ti)) for ti in thicknesses]
        else:
            sthicknesses = thicknesses
        self.thicknesses = thicknesses
        self.names = names
        self.slds = slds
        self.scaled = sthicknesses

        for ri, ti in zip(repetitions, sthicknesses):
            dsequence = sum(ti)
            if ri < 2:
                dtotal += dsequence
            elif self.show_one:
                dtotal += dsequence + dsequence / 2
            elif self.show_all or ri == 2:
                dtotal += ri * dsequence
            else:
                dtotal += 2 * dsequence + dsequence / 2
        self.dtotal = dtotal

    def get_blocks(self):
        output = []
        for ri, ti, si, cidi, ni, di in zip(
            self.repetitions, self.thicknesses, self.scaled, self.color_ids, self.names, self.slds
        ):
            infos = [LInfo(ti_j, si_j, COLORS[cid_j % len(COLORS)], ni_j, di_j) for ti_j, si_j, cid_j, ni_j, di_j in zip(ti, si, cidi, ni, di)]
            dsequence = sum(si)
            if ri < 2:
                output += infos
            elif self.show_one:
                output.append(infos + [LInfo(-ri, dsequence / 2, LColor(1.0, 1.0, 1.0), "", "")])
            elif self.show_all or ri == 2:
                output.append(ri * infos)
            else:
                output.append(infos + [LInfo(-ri, dsequence / 2, LColor(1.0, 1.0, 1.0), "", "")] + infos)
        return output


class SVGenerator:
    block_generator: BlockGenerator
    svg: svgwrite.Drawing
    use3d: bool

    unit_precision: int
    unit_nm: bool

    view_box = array([100, 200])
    move_3d = 8.0
    vanishing_point = (175.0, -15.0)
    fontsize = 10

    def __init__(
        self,
        sample: SampleBase,
        layer_names,
        rescale=True,
        show_all=False,
        show_one=False,
        show_names=False,
        show_slds=False,
        use3d=True,
        unit_precision=1,
        unit_nm=True,
        fontsize=10,
        is_xray=True,
    ):
        self.use3d = use3d
        self.fontsize = fontsize
        self.unit_precision = unit_precision
        self.unit_nm = unit_nm
        self.show_names = show_names
        self.show_slds = show_slds
        self.block_generator = BlockGenerator(
            sample.Stacks,
            layer_names=layer_names,
            is_xray=is_xray,
            rescale=rescale,
            show_all=show_all,
            show_one=show_one,
        )
        self.create_svg()

    def to_vp(self, point0):
        px, py = self.view_box * 0.01
        x0, y0 = point0
        dx = self.move_3d
        vx, vy = self.vanishing_point
        relx = dx / (vx - x0)
        dy = (vy - y0) * relx
        return ((x0 + dx) * px, (y0 + dy) * py)

    def tlabel(self, thickness):
        if self.unit_nm:
            return f"{thickness/10:.{self.unit_precision}f} nm"
        return f"{thickness:.{self.unit_precision}f} Å"

    def add_rect(self, pos: float, si: float, color: LColor = LColor(0.0, 0.0, 0.0), in_stack=False, width=90):
        px, py = self.view_box * 0.01
        if in_stack:
            x = 5 + self.fontsize
            w = width - self.fontsize
        else:
            x = 5
            w = width
        if self.use3d:
            w -= 10
            points = [
                ((x + w) * px, pos * py),
                ((x + w) * px, (pos + si) * py),
                self.to_vp(((x + w), (pos + si))),
                self.to_vp(((x + w), pos)),
            ]
            side = self.svg.polygon(points=points, stroke="black", stroke_width=0.5, fill=str(color.side))
            self.svg.add(side)
        rect = self.svg.rect((x * px, pos * py), (w * px, si * py), stroke="black", stroke_width=0.5, fill=str(color))
        self.svg.add(rect)

    def create_svg(self):
        px, py = self.view_box * 0.01
        bg = self.block_generator
        self.svg = svgwrite.Drawing(
            "genx_model.svg", size=("4cm", "8cm"), viewBox=f"0 0 {self.view_box[0]} {self.view_box[1]}"
        )
        vscale = bg.dtotal / 85.0

        blocks = bg.get_blocks()

        tc = 45 if self.use3d else 50

        pos = 90.0
        self.add_rect(pos, 100, LColor(0, 0, 0))
        bc = LColor(0, 0, 0)
        for bi in blocks:
            if isinstance(bi, list):
                block_length = sum([bij.scale / vscale for bij in bi])
                self.add_rect(pos - block_length, block_length, LColor(0.75, 0.75, 0.75), width=10 + self.fontsize)
                for bij in bi:
                    dij = bij.scale / vscale
                    pos -= dij
                    if bij.thickness < 0:
                        if self.use3d:
                            shifted = self.to_vp((85, pos + dij))
                            points = [
                                ((5 + self.fontsize) * px, (pos + dij) * py),
                                (85 * px, (pos + dij) * py),
                                shifted,
                                (shifted[0] - (85 - 5 - self.fontsize) * px, shifted[1]),
                            ]
                            side = self.svg.polygon(points=points, stroke="black", stroke_width=0.5, fill=str(bc.top))
                            self.svg.add(side)
                        paragraph = self.svg.add(self.svg.g(font_size=self.fontsize))
                        paragraph.add(
                            self.svg.text(
                                " ... ",
                                (tc * px, (pos + dij / 3.0) * py),
                                text_anchor="middle",
                                dominant_baseline="middle",
                            )
                        )
                        paragraph = self.svg.add(
                            self.svg.g(font_size=self.fontsize, transform=f"translate(5, {(pos+dij)*py}) rotate(-90)")
                        )
                        paragraph.add(
                            self.svg.text(
                                f"(x%i)" % (-bij.thickness), (0, 0), text_anchor="middle", dominant_baseline="hanging"
                            )
                        )
                    else:
                        bc = bij.color
                        self.add_rect(pos, dij, bc, in_stack=True)
                        paragraph = self.svg.add(self.svg.g(font_size=self.fontsize))
                        paragraph.add(
                            self.svg.text(
                                self.tlabel(bij.thickness),
                                (tc * px, (pos + dij / 2.0) * py),
                                text_anchor="middle",
                                dominant_baseline="middle",
                            )
                        )
                        if self.show_names:
                            paragraph.add(
                                self.svg.text(
                                    bij.name,
                                    (tc * px, (pos + dij / 2.0) * py - self.fontsize * 1.2),
                                    text_anchor="middle",
                                    dominant_baseline="middle",
                                )
                            )
                        if self.show_slds:
                            paragraph.add(
                                self.svg.text(
                                    bij.sld,
                                    (tc * px, (pos + dij / 2.0) * py + self.fontsize * 1.2),
                                    text_anchor="middle",
                                    dominant_baseline="middle",
                                )
                            )
            else:
                di = bi.scale / vscale
                pos -= di
                bc = bi.color
                self.add_rect(pos, di, bc)
                paragraph = self.svg.add(self.svg.g(font_size=self.fontsize))
                paragraph.add(
                    self.svg.text(
                        self.tlabel(bi.thickness),
                        (tc * px, (pos + di / 2.0) * py),
                        text_anchor="middle",
                        dominant_baseline="middle",
                    )
                )
                if self.show_names:
                    paragraph.add(
                        self.svg.text(
                            bi.name,
                            (tc * px, (pos + di / 2.0) * py - self.fontsize * 1.2),
                            text_anchor="middle",
                            dominant_baseline="middle",
                        )
                    )
                if self.show_slds:
                    paragraph.add(
                        self.svg.text(
                            bi.sld,
                            (tc * px, (pos + di / 2.0) * py + self.fontsize * 1.2),
                            text_anchor="middle",
                            dominant_baseline="middle",
                        )
                    )
        if self.use3d:
            shifted = self.to_vp((85, pos))
            points = [(5 * px, pos * py), (85 * px, pos * py), shifted, (shifted[0] - 80 * px, shifted[1])]
            side = self.svg.polygon(points=points, stroke="black", stroke_width=0.5, fill=str(bc.top))
            self.svg.add(side)


class SVGPanel(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self._renderer = None
        self._svg_bytes = None

    def set_svg(self, svg_text: str):
        self._svg_bytes = svg_text.encode("utf-8")
        self._renderer = QtSvg.QSvgRenderer(self._svg_bytes, self)
        self.update()

    def paintEvent(self, event):
        if not self._renderer:
            return
        painter = QtGui.QPainter(self)
        try:
            self._renderer.render(painter)
        finally:
            painter.end()


class Plugin(framework.Template):
    svg: str | None = None

    def __init__(self, parent):
        framework.Template.__init__(self, parent)
        self.parent = parent

        panel = self.NewPlotFolder("Layer Graphics")
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        self.img = SVGPanel(panel)
        layout.addWidget(self.img, 1)

        control_row = QtWidgets.QHBoxLayout()
        layout.addLayout(control_row, 0)

        left_col = QtWidgets.QVBoxLayout()
        mid_col = QtWidgets.QVBoxLayout()
        right_col = QtWidgets.QVBoxLayout()
        control_row.addLayout(left_col, 1)
        control_row.addLayout(mid_col, 1)
        control_row.addLayout(right_col, 1)

        self.rescale = QtWidgets.QCheckBox("Rescale thickness display", panel)
        self.rescale.setChecked(True)
        left_col.addWidget(self.rescale)

        self.show_all = QtWidgets.QRadioButton("All layers", panel)
        left_col.addWidget(self.show_all)
        show_topbot = QtWidgets.QRadioButton("Top/Bottom layers", panel)
        show_topbot.setChecked(True)
        left_col.addWidget(show_topbot)
        self.show_one = QtWidgets.QRadioButton("Single Repetition", panel)
        left_col.addWidget(self.show_one)

        self.show_names = QtWidgets.QCheckBox("Show Names", panel)
        left_col.addWidget(self.show_names)

        self.show_slds = QtWidgets.QCheckBox("Show SLDs", panel)
        mid_col.addWidget(self.show_slds)

        self.use3d = QtWidgets.QCheckBox("Pseudo 3d", panel)
        self.use3d.setChecked(True)
        mid_col.addWidget(self.use3d)
        self.unit_nm = QtWidgets.QCheckBox("nm-unit", panel)
        self.unit_nm.setChecked(True)
        mid_col.addWidget(self.unit_nm)

        self.unit_precision = QtWidgets.QSpinBox(panel)
        self.unit_precision.setRange(0, 5)
        self.unit_precision.setValue(1)
        mid_col.addWidget(self.unit_precision)

        self.fontsize = QtWidgets.QSpinBox(panel)
        self.fontsize.setRange(1, 20)
        self.fontsize.setValue(10)
        right_col.addWidget(self.fontsize)
        export_button = QtWidgets.QPushButton("Save to SVG...", panel)
        right_col.addWidget(export_button)
        copy_button = QtWidgets.QPushButton("Copy to Clipboard", panel)
        right_col.addWidget(copy_button)

        for ctrl in (
            self.rescale,
            self.show_names,
            self.show_slds,
            self.show_all,
            show_topbot,
            self.show_one,
            self.use3d,
            self.unit_nm,
            self.unit_precision,
            self.fontsize,
        ):
            if isinstance(ctrl, QtWidgets.QAbstractButton):
                ctrl.toggled.connect(self.OnSimulate)
            else:
                ctrl.valueChanged.connect(self.OnSimulate)

        export_button.clicked.connect(self.ExportSVG)
        copy_button.clicked.connect(self.CopyImage)

        try:
            self.OnSimulate(None)
        except Exception:
            pass

    def OnSimulate(self, _event):
        model = self.GetModel()
        if model.script_module is None:
            return
        layer_names = {
            id(item): name
            for name, item in model.script_module.__dict__.items()
            if type(item).__name__ == "Layer"
        }
        gen = SVGenerator(
            model.script_module.sample,
            layer_names=layer_names,
            show_names=self.show_names.isChecked(),
            show_slds=self.show_slds.isChecked(),
            rescale=self.rescale.isChecked(),
            show_all=self.show_all.isChecked(),
            show_one=self.show_one.isChecked(),
            use3d=self.use3d.isChecked(),
            unit_nm=self.unit_nm.isChecked(),
            unit_precision=int(self.unit_precision.value()),
            fontsize=self.fontsize.value(),
            is_xray=model.script_module.inst.probe == "x-ray",
        )
        self.svg = gen.svg.tostring()
        self.img.set_svg(self.svg)

    def ExportSVG(self):
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.img,
            "Save layer sketch image",
            "",
            "SVG Image (*.svg)",
        )
        if not fname:
            return
        with open(fname, "w", encoding="utf-8") as fh:
            fh.write(self.svg or "")

    def CopyImage(self):
        if not self.svg:
            return
        renderer = QtSvg.QSvgRenderer(self.svg.encode("utf-8"))
        image = QtGui.QImage(300, 600, QtGui.QImage.Format.Format_ARGB32)
        image.fill(QtCore.Qt.GlobalColor.white)
        painter = QtGui.QPainter(image)
        try:
            renderer.render(painter)
        finally:
            painter.end()
        QtWidgets.QApplication.clipboard().setImage(image)
