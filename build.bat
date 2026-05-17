@echo off
REM ================================================================
REM build.bat  --  Compile islpstat C library to a Windows DLL
REM ================================================================
REM
REM Requirements:
REM   MinGW-w64 with gcc on PATH  (e.g. from MSYS2: pacman -S mingw-w64-x86_64-gcc)
REM   or TDM-GCC  (https://jmeubank.github.io/tdm-gcc/)
REM
REM Run this once from the islpstat/ root directory:
REM     build.bat
REM
REM This produces:  islpstat\islpstat.dll
REM The Python wrapper (python/_core.py) loads it automatically.
REM ================================================================

echo Building islpstat.dll ...

gcc -O2 -shared -o islpstat.dll ^
    src/stat_dist.c ^
    src/linalg.c ^
    src/linreg.c ^
    src/logreg.c ^
    src/lda.c ^
    src/classify.c ^
    -lm ^
    -Wl,--export-all-symbols

IF %ERRORLEVEL% EQU 0 (
    echo.
    echo SUCCESS: islpstat.dll built.
    echo.
    echo To verify, run from the islpstat/ directory:
    echo     python tests\test_vs_statsmodels.py
) ELSE (
    echo.
    echo FAILED. Make sure MinGW gcc is on your PATH.
    echo Install via MSYS2:  pacman -S mingw-w64-x86_64-gcc
)
