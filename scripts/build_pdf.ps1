param([string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot))
$ErrorActionPreference = 'Stop'
$html = Join-Path $ProjectRoot 'output\final_assignment.html'
$pdf = Join-Path $ProjectRoot 'output\final_assignment.pdf'
$edge = 'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
if (-not (Test-Path $html)) { throw "HTML source not found: $html" }
if (-not (Test-Path $edge)) { throw "Microsoft Edge not found at $edge" }
$fullHtml = (Resolve-Path $html).Path
$uri = [System.Uri]::new($fullHtml).AbsoluteUri
$args = @('--headless=new','--disable-gpu','--no-first-run','--no-pdf-header-footer',"--print-to-pdf=$pdf",$uri)
$proc = Start-Process -FilePath $edge -ArgumentList $args -WindowStyle Hidden -PassThru
$proc | Wait-Process -Timeout 90
if (-not (Test-Path $pdf)) { throw 'PDF export did not create the expected file.' }
Get-Item $pdf | Select-Object FullName,Length
