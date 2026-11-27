@echo off
chcp 65001 >nul
title FluxoVC
cd /d "%~dp0"

rem Inicializador do FluxoVC para Windows: na primeira vez cria o ambiente e instala as dependencias.

set "PY=python"
where py >nul 2>nul && set "PY=py -3"

if exist ".venv\.instalado" goto iniciar

echo.
echo  Primeira execucao: criando o ambiente e instalando as dependencias.
echo  Isso leva de 5 a 15 minutos e precisa de internet. Nas proximas vezes abre direto.
echo.
if not exist ".venv\Scripts\python.exe" (
    %PY% -m venv .venv
    if errorlevel 1 goto sem_python
)
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto erro_pip
echo ok> ".venv\.instalado"

:iniciar
echo.
echo  Abrindo o FluxoVC em http://localhost:8501 ... feche esta janela para encerrar.
echo.
".venv\Scripts\python.exe" -m streamlit run app.py
goto fim

:sem_python
echo.
echo  ERRO: Python 3.10 ou mais novo nao encontrado.
echo  Instale em https://www.python.org/downloads/ marcando "Add python.exe to PATH" e rode de novo.
pause
goto fim

:erro_pip
echo.
echo  ERRO: a instalacao das dependencias falhou.
echo  Se a mensagem citar "Long Path", mova a pasta do projeto para um caminho curto, como C:\fluxovc,
echo  apague a pasta .venv e rode este arquivo de novo.
pause

:fim
