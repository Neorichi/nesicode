; Inno Setup script: builds dist\nesicode-<version>-windows-x64-setup.exe from dist\nesicode\ (PyInstaller)
; Per-user install: no administrator rights needed.
#ifndef AppVersion
  #define AppVersion "0.4.0"
#endif

[Setup]
AppId={{7E3B6F2A-4C1D-4E8A-9B7F-2D5A1C3E9F10}
AppName=Nesicode
AppVersion={#AppVersion}
AppPublisher=Nesi Labs
AppPublisherURL=https://github.com/Neorichi/nesicode
DefaultDirName={localappdata}\Programs\Nesicode
DefaultGroupName=Nesicode
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\..\dist
OutputBaseFilename=Nesicode-{#AppVersion}-windows-x64-setup
SetupIconFile=..\icons\nesicode.ico
UninstallDisplayIcon={app}\Nesicode.exe
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\..\dist\nesicode\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Nesicode"; Filename: "{app}\Nesicode.exe"
Name: "{autodesktop}\Nesicode"; Filename: "{app}\Nesicode.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\Nesicode.exe"; Description: "{cm:LaunchProgram,Nesicode}"; Flags: nowait postinstall skipifsilent

; User data (%APPDATA%\nesicode) is kept on uninstall on purpose.
