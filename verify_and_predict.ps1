$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$assignmentPython = (Get-Command python -ErrorAction Stop).Source
foreach ($assignmentVariant in @(1, 2)) {
    & $assignmentPython src/predict.py --variant $assignmentVariant --sample sample_submission.csv
    if ($LASTEXITCODE -ne 0) { throw "Prediction failed for variant $assignmentVariant" }
}
& $assignmentPython src/verify.py
if ($LASTEXITCODE -ne 0) { throw 'Output verification failed' }
Write-Output 'Both prediction files were regenerated and verified.'
