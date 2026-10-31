; Inno Setup script for the AOE4 Smart Overlay (per-user install).
;
; Pass the portable release root with: ISCC /DSourceDir=<path>
; SourceDir must contain: overlay\aoe4_overlay.exe, docs\, Readme.md, etc.
;
; App version: fixed 2.16.0 baseline is used below. If, in a future revision,
; the version is derived from version.json, document that mechanism here; for
; now the fixed value matches version.json in the repository root.

#ifndef SourceDir
  #error SourceDir is not defined. Build with: ISCC /DSourceDir=<path to extracted portable release>
#endif

#define MyAppName "AOE4 Smart Overlay"
#define MyAppVersion "2.16.0"
#define MyAppPublisher "RTS Overlay"
#define MyAppURL "https://github.com/CraftySalamander/RTS_Overlay"
#define MyAppExeName "aoe4_overlay.exe"
#define MyAppId "CraftySalamander.RTS_Overlay.AOE4SmartOverlay"

[Setup]
AppId={#MyAppId}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={localappdata}\Programs\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=Output
OutputBaseFilename=aoe4-smart-overlay-setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayName={#MyAppName}
UninstallDisplayIcon={app}\overlay\{#MyAppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\overlay\{#MyAppExeName}"; WorkingDir: "{app}\overlay"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{userdesktop}\{#MyAppName}"; Filename: "{app}\overlay\{#MyAppExeName}"; WorkingDir: "{app}\overlay"; Tasks: desktopicon

[Run]
Filename: "{app}\overlay\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; WorkingDir: "{app}\overlay"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}"

