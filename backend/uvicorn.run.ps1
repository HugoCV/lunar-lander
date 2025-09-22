param(
    [int]$Port = 8000,
    [string]$Host = "0.0.0.0",
    [switch]$Reload
)

# Ajustar el PYTHONPATH a la carpeta actual (para que los imports funcionen bien)
$env:PYTHONPATH = "$PSScriptRoot"

# Construir los flags de uvicorn
$reloadFlag = ""
if ($Reload) {
    $reloadFlag = "--reload"
}

Write-Host "🚀 Iniciando FastAPI con Uvicorn en http://$Host:$Port"

uvicorn app.main:app $reloadFlag --host $Host --port $Port
