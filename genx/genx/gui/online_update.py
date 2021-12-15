"""
Funcitoniality for checking online for new versions, prompting the user and downloading potential updates.
"""

import os
import sys
import requests
import base64
import tempfile
import subprocess
import wx
import wx.html as html

from logging import debug
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from .help import rst_html
from .output_dialog import TextOutputDialog
from ..version import __version__ as GENX_VERSION

GITHUB_URL="https://api.github.com/repos/aglavic/genx/releases/latest"
GITHUB_TAGS="https://api.github.com/repos/aglavic/genx/tags"

retry_strategy = Retry(
    total=3,
    status_forcelist=[429, 500, 502, 503, 504],
    method_whitelist=["HEAD", "GET", "OPTIONS"]
)

def check_version():
    adapter = HTTPAdapter(max_retries=retry_strategy)
    http = requests.Session()
    http.mount("https://", adapter)

    response = http.get(GITHUB_URL).json()
    if not 'name' in response:
        debug(f"Response from GitHub unexpected: {response!r}")
        return True
    version=response["name"]
    return version[1:]==GENX_VERSION

class VersionInfoDialog(wx.Dialog):
    def __init__(self, parent):
        wx.Dialog.__init__(self, parent, style=wx.DEFAULT_DIALOG_STYLE|wx.RESIZE_BORDER,
                           name=f'GenX update')
        self.SetTitle(f'GenX update available')
        self.SetSize(wx.Size(800, 800))
        info=self.collect_release_info()
        changes=self.filter_readme(info['readme_text'])

        vbox=wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(vbox)
        vbox.Add(wx.StaticText(self, label=f'Newest Release: {info["version"]}'))
        vbox.Add(wx.StaticText(self, label=f'Date: {info["date"]}'))
        vbox.Add(wx.StaticText(self, label=f'\nChanges since {GENX_VERSION}'))

        self.html_win=html.HtmlWindow(self, -1,
                                      style=wx.NO_FULL_REPAINT_ON_RESIZE)
        vbox.Add(self.html_win, 1, flag=wx.EXPAND, border=20)
        self.html_win.SetPage(rst_html(changes))

        btn_box=wx.BoxSizer(wx.HORIZONTAL)
        vbox.Add(btn_box)
        self.download_link=None
        if sys.platform.startswith('win') and info['setup_file']:
            # assume Windows users have installed from setup
            self.download_link=info['setup_file']
            self.download_size=info['setup_file_size']
            btn=wx.Button(self, label='Download and Install...')
            btn_box.Add(btn)
            self.Bind(wx.EVT_BUTTON, self.download_and_install, btn)
        if self.check_pip_version():
            btn=wx.Button(self, label='Pip in-place update...')
            btn_box.Add(btn)
            self.Bind(wx.EVT_BUTTON, self.update_pip, btn)

    def check_pip_version(self):
        try:
            import pkg_resources
        except ImportError:
            return False
        pkg_names=[si.key for si in pkg_resources.working_set]
        pkg_versions=[si.version for si in pkg_resources.working_set]
        if 'genx3' in pkg_names and pkg_versions[pkg_names.index('genx3')]==GENX_VERSION:
            return True
        return False

    def download_and_install(self, evt):
        adapter = HTTPAdapter(max_retries=retry_strategy)
        http = requests.Session()
        http.mount("https://", adapter)
        debug(f'Downloading file {self.download_link}')
        dia=wx.ProgressDialog('Downloading GenX...', 'The latest GenX setup is being downloaded from Github.\n'
                                                     'The setup will start once the download is completed.',
                              maximum=100, parent=None, style=wx.PD_APP_MODAL|wx.PD_AUTO_HIDE)
        dia.Show()
        with http.get(self.download_link, allow_redirects=True, stream=True) as res:
            res.raise_for_status()
            with tempfile.NamedTemporaryFile('wb', delete=False, suffix='.exe', prefix='genx_setup_') as tmp:
                debug(f'Save to temporary file f{tmp.name}')
                s=0
                for chunk in res.iter_content(chunk_size=8192):
                    s+=len(chunk)
                    rel_step=s/self.download_size
                    dia.Update(int(rel_step*100))
                    tmp.write(chunk)
        debug(f'Calling setup file f{tmp.name}')
        dia.Destroy()
        subprocess.Popen(tmp.name)
        debug(f'Started process f{tmp.name}')
        self.EndModal(wx.ID_DELETE)

    def update_pip(self, evt):
        dia=TextOutputDialog(self)
        dia.SetSize(wx.Size(800,500))
        dia.Show()
        dia.write('Starting "pip install genx3 --upgrade":\n\n')
        with subprocess.Popen([sys.executable, '-m', 'pip', 'install', 'genx3', '--upgrade'],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as proc:
            while proc.poll() is None:
                dia.write(proc.stdout.readline().decode('utf-8'))
        dia.write(f'\n\nProcess finished with exit code {proc.poll()}')
        wx.Yield()
        dia.ShowModal()
        dia.Destroy()
        if proc.poll()==0:
            self.EndModal(wx.ID_OK)
        else:
            self.EndModal(wx.ID_CANCEL)

    @staticmethod
    def get_file(http, tree, path):
        path_items = path.split('/')
        for pi in path_items[:-1]:
            for ti in tree:
                if ti['path'] == pi:
                    tree = http.get(ti['url']).json()['tree']
                    break
        for ti in tree:
            if ti['path'] == path_items[-1]:
                data=http.get(ti['url']).json()['content']
                return base64.b64decode(data).decode('utf-8')

    @staticmethod
    def collect_release_info():
        adapter = HTTPAdapter(max_retries=retry_strategy)
        http = requests.Session()
        http.mount("https://", adapter)

        response = http.get(GITHUB_URL).json()
        output=dict(date=response['published_at'], version=response['name'][1:],
                    setup_file=None, deb_file=None)
        for asset in response['assets']:
            if asset['name'].endswith('.exe'):
                output['setup_file']=asset['browser_download_url']
                output['setup_file_size']=asset['size']
            elif asset['name'].endswith('py38.deb'):
                output['deb_file']=asset['browser_download_url']
        tags = requests.get(GITHUB_TAGS).json()
        tag = tags[[ti['name'] for ti in tags].index(response['tag_name'])]
        commit = http.get(tag['commit']['url']).json()
        ctmp = http.get(commit['url']).json()
        tree = http.get(ctmp['commit']['tree']['url']).json()
        output['readme_text'] = VersionInfoDialog.get_file(http, tree['tree'], 'genx/README.txt')
        return output

    @staticmethod
    def filter_readme(txt):
        # shorten the readme to only show changes since this version
        start = txt.find("Change")
        end=txt.find(GENX_VERSION)
        txt=txt[start:end]
        return txt.rsplit('\n', 1)[0]
