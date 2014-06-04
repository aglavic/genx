; Script generated by the Inno Setup Script Wizard.
; SEE THE DOCUMENTATION FOR DETAILS ON CREATING INNO SETUP SCRIPT FILES!

[Setup]
; NOTE: The value of AppId uniquely identifies this application.
; Do not use the same AppId value in installers for other applications.
; (To generate a new GUID, click Tools | Generate GUID inside the IDE.)
AppId={{03439AE4-FE61-49AC-8D3F-1351147CE8FC}
AppName=GenX
AppVerName=GenX 2.2.0
AppPublisher=Matts Bjorck
AppPublisherURL=https://sourceforge.net/projects/genx
AppSupportURL=https://sourceforge.net/projects/genx
AppUpdatesURL=https://sourceforge.net/projects/genx
DefaultDirName={pf}\GenX
DefaultGroupName=GenX      
AllowNoIcons=true
OutputDir=Z:\Documents\GenX\sf-code\tags\v2.2.0\windows_build
OutputBaseFilename=install_genx_win32
Compression=lzma
SolidCompression=true
ChangesAssociations=true
PrivilegesRequired=admin
UsePreviousTasks=yes
WizardImageFile=..\windows_build\install_wizard_bkg.bmp
WizardSmallImageFile=..\windows_build\install_wizard_small.bmp
InfoBeforeFile=..\README.txt


[Languages]
Name: english; MessagesFile: compiler:Default.isl

[Files]
Source: ..\dist\genx.exe; DestDir: {app}; Flags: ignoreversion
Source: ..\dist\*.*; DestDir: {app}; Flags: ignoreversion recursesubdirs

[Icons]
Name: {group}\GenX; Filename: {app}\genx.exe; IconFilename: {app}\genx.exe; IconIndex: 0
Name: {group}\{cm:UninstallProgram,GenX}; Filename: {uninstallexe}

[Registry]
Root: HKCU; Subkey: Software\Classes\.gx; ValueType: string; ValueName: ; ValueData: GenX; Tasks: associate; Flags: uninsdeletevalue createvalueifdoesntexist
Root: HKCU; Subkey: Software\Classes\GenX; ValueType: string; ValueName: ; ValueData: GenX model; Tasks: associate; Flags: uninsdeletekey createvalueifdoesntexist
Root: HKCU; Subkey: Software\Classes\GenX\DefaultIcon; ValueType: string; ValueName: ; ValueData: {app}\genx.exe,1; Tasks: associate; Flags: createvalueifdoesntexist
Root: HKCU; Subkey: Software\Classes\GenX\shell\open\command; ValueType: string; ValueName: ; ValueData: """{app}\genx.exe"" ""%1"""; Tasks: associate; Flags: createvalueifdoesntexist

[Run]

[Tasks]
Name: associate; Description: Create registry entries for file association; GroupDescription: Associate Filetypes:; Flags: 
