; Script generated by the Inno Setup Script Wizard.
; SEE THE DOCUMENTATION FOR DETAILS ON CREATING INNO SETUP SCRIPT FILES!

[Setup]
; NOTE: The value of AppId uniquely identifies this application.
; Do not use the same AppId value in installers for other applications.
; (To generate a new GUID, click Tools | Generate GUID inside the IDE.)
AppId={{54F9D347-19F3-4833-BA3B-154B6BBED1F9}
AppName=GenX 3
AppVerName=GenX {version}
AppPublisher=Artur Glavic
AppPublisherURL=https://sourceforge.net/projects/genx
AppSupportURL=https://sourceforge.net/projects/genx
AppUpdatesURL=https://sourceforge.net/projects/genx
DefaultDirName={autopf}\GenX 3
DefaultGroupName=GenX 3
AllowNoIcons=true

OutputBaseFilename=GenX-{version}_win64_setup
Compression=lzma
SolidCompression=true
ChangesAssociations=true
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
UsePreviousTasks=yes
WizardImageFile=.\install_wizard_bkg.bmp
WizardSmallImageFile=.\install_wizard_small.bmp
WizardStyle=modern
InfoBeforeFile=..\README.txt
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible


[Languages]
Name: english; MessagesFile: compiler:Default.isl

[Files]
Source: ..\dist\genx\genx.exe; DestDir: {app}; Flags: ignoreversion
Source: ..\dist\genx\*.*; DestDir: {app}; Flags: ignoreversion recursesubdirs

[Icons]
Name: {group}\GenX 3; Filename: {app}\genx.exe; IconFilename: {app}\genx.exe; IconIndex: 0
Name: {group}\{cm:UninstallProgram,GenX 3}; Filename: {uninstallexe}

[Registry]
Root: HKA; Subkey: Software\Classes\.gx; ValueType: string; ValueName: ; ValueData: GenX; Tasks: associate; Flags: uninsdeletevalue createvalueifdoesntexist
Root: HKA; Subkey: Software\Classes\.hgx; ValueType: string; ValueName: ; ValueData: GenX; Tasks: associate; Flags: uninsdeletevalue createvalueifdoesntexist
Root: HKA; Subkey: Software\Classes\.ort; ValueType: string; ValueName: ; ValueData: GenX; Tasks: associate_ort; Flags: uninsdeletevalue createvalueifdoesntexist
Root: HKA; Subkey: Software\Classes\GenX; ValueType: string; ValueName: ; ValueData: GenX model; Tasks: associate; Flags: uninsdeletekey createvalueifdoesntexist
Root: HKA; Subkey: Software\Classes\GenX\DefaultIcon; ValueType: string; ValueName: ; ValueData: {app}\genx.exe,1; Tasks: associate; Flags: createvalueifdoesntexist
Root: HKA; Subkey: Software\Classes\GenX\shell\open\command; ValueType: string; ValueName: ; ValueData: """{app}\genx.exe"" ""%1"""; Tasks: associate; Flags: createvalueifdoesntexist

[Run]
Filename: "{app}\genx_console.exe"; Parameters: "--compile-nb"; Description: "Pre-compile JIT functions"; StatusMsg: "Pre-compiling JIT functions..."; Tasks: compile_jit; Flags: runasoriginaluser

[Tasks]
Name: associate; Description: Create registry entries for file association; GroupDescription: Associate Filetypes:; Flags: 
Name: associate_ort; Description: Let GenX open ORSO files (*.ort); GroupDescription: Associate Filetypes:; Flags: unchecked
Name: compile_jit; Description: Pre-compile JIT functions; GroupDescription: After Installation:; Flags:
Name: uninstall; Description: Uninstall previous version of GenX 3; GroupDescription: Remove Old Files:; Flags: 

; Perform uninstall before installing new version, just in case there are conflicts
[Code]
function GetUninstallerExePath(): String;
var
    UninstallerExePath: String;
    INSTALL_KEY: String;
begin
    INSTALL_KEY := ExpandConstant('Software\Microsoft\Windows\CurrentVersion\Uninstall\{#SetupSetting("AppId")}_is1');
    Result := '';
    if RegQueryStringValue(HKEY_AUTO, INSTALL_KEY, 'UninstallString', UninstallerExePath) then
    begin
        Result := UninstallerExePath;
    end
    else
    begin
        Log('Uninstaller location not found in the registry. ('+INSTALL_KEY+')');
    end;
end;

procedure UninstallPreviousVersion();
var
    UninstallerExePath: String;
    ResultCode: Integer;
begin
    Log('UninstallPreviousVersion called');
    UninstallerExePath := GetUninstallerExePath();
    if ((UninstallerExePath <> '') and WizardIsTaskSelected('uninstall')) then
    begin
        Log('Executing uninstaller: ' + UninstallerExePath);
        if ShellExec('runas', UninstallerExePath, '/NORESTART /SILENT', '', SW_SHOWNORMAL, ewWaitUntilTerminated, ResultCode) then
        begin
            // MsgBox('Previous version uninstalled successfully.', mbInformation, MB_OK);
        end
        else
        begin
            Log('Failed to uninstall the previous version. ResultCode: ' + IntToStr(ResultCode));
            MsgBox('Failed to uninstall the previous version. ResultCode: ' + IntToStr(ResultCode), mbError, MB_OK);
        end;
    end;
end;


procedure CurStepChanged(CurStep: TSetupStep);
begin
    if (CurStep = ssInstall) then
    begin
        UninstallPreviousVersion();
    end;
end;
