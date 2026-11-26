@echo off
rem Builds smoovplay-camera.dll (SmoovPlay Camera, the DirectShow source OBS opens) with Visual Studio 2019+ Build
rem Tools: vcam.cpp and Microsoft's DirectShow base classes (baseclasses\, MIT). 64-bit only.
rem   build.bat                   -> build\smoovplay-camera.dll
rem Register it for the current user (no admin):  regsvr32 /s /n /i:user build\smoovplay-camera.dll
rem and remove it again:                           regsvr32 /s /u /n /i:user build\smoovplay-camera.dll
setlocal
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath > "%TEMP%\vcam_vsdir.txt"
set /p VSDIR=<"%TEMP%\vcam_vsdir.txt"
del "%TEMP%\vcam_vsdir.txt"
if "%VSDIR%"=="" (
  echo Visual Studio C++ build tools not found
  exit /b 1
)
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cd /d "%~dp0"
if not exist build mkdir build
set BASE=baseclasses\amextra.cpp baseclasses\amfilter.cpp baseclasses\amvideo.cpp baseclasses\arithutil.cpp ^
  baseclasses\combase.cpp baseclasses\ctlutil.cpp baseclasses\dllentry.cpp ^
  baseclasses\dllsetup.cpp baseclasses\mtype.cpp baseclasses\pstream.cpp baseclasses\refclock.cpp ^
  baseclasses\schedule.cpp baseclasses\source.cpp baseclasses\sysclock.cpp baseclasses\wxdebug.cpp ^
  baseclasses\wxlist.cpp baseclasses\wxutil.cpp
cl /nologo /LD /MT /O2 /EHsc /W3 /DUNICODE /D_UNICODE /DWIN32 /D_WINDOWS /DNDEBUG /Ibaseclasses ^
   vcam.cpp %BASE% /Fobuild\ /Febuild\smoovplay-camera.dll ^
   /link /DEF:vcam.def strmiids.lib winmm.lib ole32.lib oleaut32.lib advapi32.lib user32.lib uuid.lib gdi32.lib || exit /b 1
echo built %~dp0build\smoovplay-camera.dll
