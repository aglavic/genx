"""
Qt functionality for checking online for new versions, prompting the user and downloading updates.
"""

import base64
import os
import subprocess
import sys
import tempfile
from logging import debug

import requests
from PySide6 import QtCore, QtGui, QtWidgets
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from ..version import __version__ as GENX_VERSION

GITHUB_URL = "https://api.github.com/repos/aglavic/genx/releases/latest"
GITHUB_TAGS = "https://api.github.com/repos/aglavic/genx/tags"


def _get_Retry_kw():
    if hasattr(Retry.DEFAULT, "allowed_methods"):
        return {"allowed_methods": ["HEAD", "GET", "OPTIONS"]}
    return {"method_whitelist": ["HEAD", "GET", "OPTIONS"]}


retry_strategy = Retry(total=3, status_forcelist=[429, 500, 502, 503, 504], **_get_Retry_kw())


def rst_html(text):
    try:
        from docutils.core import publish_doctree, publish_from_doctree
        from docutils.parsers.rst import roles
    except ImportError:
        return "For proper display install docutils.<br>\n" + text.replace("\n", "<br>\n")

    def _role_fn(name, rawtext, text, lineno, inliner, options=None, content=None):
        return [], []

    roles.register_canonical_role("mod", _role_fn)
    return publish_from_doctree(publish_doctree(text), writer_name="html").decode()


def check_version():
    adapter = HTTPAdapter(max_retries=retry_strategy)
    http = requests.Session()
    http.mount("https://", adapter)

    try:
        response = http.get(GITHUB_URL).json()
    except Exception:
        debug("Could not check for update, error in request:", exc_info=True)
        return True
    if "name" not in response:
        debug(f"Response from GitHub unexpected: {response!r}")
        return True
    version = response["name"]
    return version[1:] == GENX_VERSION


class TextOutputDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, title="Command Output"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(800, 500)

        self.text = QtWidgets.QPlainTextEdit(self)
        self.text.setReadOnly(True)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close, self)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.text, 1)
        layout.addWidget(buttons, 0)

    def write(self, text: str):
        self.text.moveCursor(QtGui.QTextCursor.MoveOperation.End)
        self.text.insertPlainText(text)
        self.text.moveCursor(QtGui.QTextCursor.MoveOperation.End)


class VersionInfoDialog(QtWidgets.QDialog):
    RESULT_RESTART = 1
    RESULT_QUIT = 2

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("GenX update available")
        self.resize(800, 800)

        info = self.collect_release_info()
        changes = self.filter_readme(info["readme_text"])

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel(f'Newest Release: {info["version"]}'))
        layout.addWidget(QtWidgets.QLabel(f'Date: {info["date"]}'))
        layout.addWidget(QtWidgets.QLabel(f"\nChanges since {GENX_VERSION}"))

        self.html_win = QtWidgets.QTextBrowser(self)
        self.html_win.setHtml(rst_html(changes))
        layout.addWidget(self.html_win, 1)

        button_row = QtWidgets.QHBoxLayout()
        layout.addLayout(button_row)
        self.download_link = None
        if sys.platform.startswith("win") and info.get("setup_file"):
            self.download_link = info["setup_file"]
            self.download_size = info["setup_file_size"]
            btn = QtWidgets.QPushButton("Download and Install...")
            btn.clicked.connect(self.download_and_install)
            button_row.addWidget(btn)
        if self.check_pip_version():
            btn = QtWidgets.QPushButton("Pip in-place update...")
            btn.clicked.connect(self.update_pip)
            button_row.addWidget(btn)
        button_row.addStretch(1)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close, self)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def check_pip_version(self):
        try:
            import pkg_resources
        except ImportError:
            return False
        pkg_names = [si.key for si in pkg_resources.working_set]
        pkg_versions = [si.version for si in pkg_resources.working_set]
        return "genx3" in pkg_names and pkg_versions[pkg_names.index("genx3")] == GENX_VERSION

    def download_and_install(self):
        adapter = HTTPAdapter(max_retries=retry_strategy)
        http = requests.Session()
        http.mount("https://", adapter)
        debug(f"Downloading file {self.download_link}")
        dia = QtWidgets.QProgressDialog(
            "The latest GenX setup is being downloaded from Github.\n"
            "The setup will start once the download is completed.",
            "Cancel",
            0,
            100,
            self,
        )
        dia.setWindowTitle("Downloading GenX...")
        dia.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        dia.show()
        with http.get(self.download_link, allow_redirects=True, stream=True) as res:
            res.raise_for_status()
            with tempfile.NamedTemporaryFile("wb", delete=False, suffix=".exe", prefix="genx_setup_") as tmp:
                debug(f"Save to temporary file {tmp.name}")
                s = 0
                for chunk in res.iter_content(chunk_size=8192):
                    if dia.wasCanceled():
                        dia.close()
                        return
                    s += len(chunk)
                    rel_step = s / self.download_size
                    dia.setValue(int(rel_step * 100))
                    QtWidgets.QApplication.processEvents()
                    tmp.write(chunk)
        debug(f"Calling setup file {tmp.name}")
        dia.close()
        subprocess.Popen(tmp.name)
        debug(f"Started process {tmp.name}")
        self.done(self.RESULT_QUIT)

    def update_pip(self):
        dia = TextOutputDialog(self, title="Pip Update")
        dia.show()
        dia.write('Starting "pip install genx3 --upgrade":\n\n')
        with subprocess.Popen(
            [sys.executable, "-m", "pip", "install", "genx3", "--upgrade"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ) as proc:
            while proc.poll() is None:
                line = proc.stdout.readline()
                if not line:
                    break
                dia.write(line.decode("utf-8", errors="replace"))
                QtWidgets.QApplication.processEvents()
        dia.write(f"\n\nProcess finished with exit code {proc.poll()}")
        dia.exec()
        if proc.poll() == 0:
            self.done(self.RESULT_RESTART)
        else:
            self.reject()

    @staticmethod
    def get_file(http, tree, path):
        path_items = path.split("/")
        for pi in path_items[:-1]:
            for ti in tree:
                if ti["path"] == pi:
                    tree = http.get(ti["url"]).json()["tree"]
                    break
        for ti in tree:
            if ti["path"] == path_items[-1]:
                data = http.get(ti["url"]).json()["content"]
                return base64.b64decode(data).decode("utf-8")

    @staticmethod
    def collect_release_info():
        adapter = HTTPAdapter(max_retries=retry_strategy)
        http = requests.Session()
        http.mount("https://", adapter)

        response = http.get(GITHUB_URL).json()
        output = dict(date=response["published_at"], version=response["name"][1:], setup_file=None, deb_file=None)
        for asset in response["assets"]:
            if asset["name"].endswith(".exe"):
                output["setup_file"] = asset["browser_download_url"]
                output["setup_file_size"] = asset["size"]
            elif asset["name"].endswith("py38.deb"):
                output["deb_file"] = asset["browser_download_url"]
        tags = requests.get(GITHUB_TAGS).json()
        tag = tags[[ti["name"] for ti in tags].index(response["tag_name"])]
        commit = http.get(tag["commit"]["url"]).json()
        ctmp = http.get(commit["url"]).json()
        tree = http.get(ctmp["commit"]["tree"]["url"]).json()
        output["readme_text"] = VersionInfoDialog.get_file(http, tree["tree"], "genx/README.txt")
        return output

    @staticmethod
    def filter_readme(txt):
        start = txt.find("Change")
        end = txt.find(GENX_VERSION)
        txt = txt[start:end]
        return txt.rsplit("\n", 1)[0]
