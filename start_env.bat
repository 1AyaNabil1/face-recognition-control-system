@echo off
echo [INFO] Activating Virtual Environment...
if not exist venv (
    echo [INFO] Creating Virtual Environment...
    python -m venv venv
)
call venv\Scripts\activate
echo [✔] Environment Ready!
cmd /k
