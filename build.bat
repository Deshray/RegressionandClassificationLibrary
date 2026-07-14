@echo off
REM islpstat v2 — build script
REM Run from the islpstat_v2\ directory.

set SRC=src\stat_dist.c src\linalg.c src\linreg.c src\logreg.c ^
        src\lda.c src\classify.c ^
        src\linreg_diag.c src\classify_ext.c ^
        src\knn.c src\poisson.c src\multilogreg.c

echo Building islpstat.dll ...
gcc -O2 -shared -o islpstat.dll %SRC% -lm -Wall
if %errorlevel% neq 0 (
    echo BUILD FAILED
    exit /b 1
)
echo BUILD OK  →  islpstat.dll
