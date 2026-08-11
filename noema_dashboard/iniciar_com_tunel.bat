@echo off
rem Noema Dashboard + tunel Cloudflare em um clique.
rem Requisitos (uma vez): winget install Cloudflare.cloudflared
rem A URL publica aparece abaixo (https://....trycloudflare.com) — muda a cada execucao.
rem AVISO: o dashboard nao tem senha; nao compartilhe a URL e feche esta janela ao terminar.

cd /d "%~dp0"
start "Noema Dashboard" "..\noema_env\Scripts\python.exe" servidor.py
timeout /t 3 /nobreak >nul
cloudflared tunnel --url http://localhost:7860
