@echo off
setlocal ENABLEDELAYEDEXPANSION

set CFG=%1
if "%CFG%"=="" set CFG=config.yaml

echo [run_all] Using config: %CFG%

:: 1) Fetch data (prefer Python fetcher; fallback to bash if available)
python scripts/fetch_uniprot_ec.py
if errorlevel 1 (
  echo [run_all] Python fetcher failed, trying bash variant...
  bash -c "exit" >nul 2>&1
  if %errorlevel%==0 (
    bash scripts/fetch_uniprot_ec.sh
    if errorlevel 1 goto :error
  ) else (
    echo [run_all] No bash available and python fetch failed.
    goto :error
  )
)

:: 1.5) Targeted TrEMBL top-up (build merged joined TSV)
echo [run_all] Top-up: SwissProt + targeted TrEMBL
python scripts/topup_trembl_targeted.py --config %CFG% --target-min auto --buffer 2 --cap-per-ec 50 --max-ecs 2000
if errorlevel 1 (
  echo [run_all] [warn] Top-up step failed; proceeding with Swiss-Prot only
)

:: 2) Prepare splits
python -m src.prepare_split -c %CFG%
if errorlevel 1 goto :error

:: 2.1) Augment train split now that splits exist
python scripts/topup_trembl_targeted.py --config %CFG% --target-min auto --buffer 2 --augment-train --augment-only
if errorlevel 1 (
  echo [run_all] [warn] Train augmentation step failed; continuing
)

:: 2.5) Cluster sequences for identity-aware sampling
echo [run_all] Clustering sequences for identity-aware sampling
python scripts/cluster_sequences.py -c %CFG%
if errorlevel 1 goto :error

:: 3) Embed sequences
python -m src.embed_sequences -c %CFG%
if errorlevel 1 goto :error

:: 4) Train ProtoNet
python -m src.train_protonet -c %CFG%
if errorlevel 1 goto :error

:: 5) Evaluate
python -m src.eval_protonet -c %CFG%
if errorlevel 1 goto :error

echo [run_all] Done. Check results\ for outputs.
exit /b 0

:error
echo [run_all] An error occurred. Please review the messages above.
exit /b 1
