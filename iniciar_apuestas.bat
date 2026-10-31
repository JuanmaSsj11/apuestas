@echo off
title Modelo Predictivo Cuantitativo de Apuestas Deportivas (+EV)
chcp 65001 >nul
cd /d "c:\Users\ssjju\Downloads\apuestas"

echo ===============================================================================
echo            MODELO PREDICTIVO CUANTITATIVO DE APUESTAS DEPORTIVAS (+EV)
echo ===============================================================================
echo.
echo  Abriendo Panel Cuantitativo...
echo  Acceso en PC:      http://localhost:8501
echo  Acceso en Celular: http://192.168.1.84:8501 (mismo Wi-Fi)
echo.

python -m streamlit run app_ui.py --server.address 0.0.0.0 --server.port 8501 --server.headless false

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Hubo un problema al iniciar Streamlit, ejecutando en consola...
    python run_pipeline.py
    pause
)
