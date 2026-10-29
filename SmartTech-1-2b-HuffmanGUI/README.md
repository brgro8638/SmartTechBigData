## 1. sett opp venv

```powershell
python -m venv venv
```
## 2. aktiver venv

```powershell
venv\Scripts\Activate
```
dersom PowerShell sier: `cannot be loaded because running scripts is disabled on this system`

```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
```
forsøk deretter å aktivere venv på nytt.
## 3. installer dependencies

```powershell
pip install -r requirements.txt
```
## 4. kjør

```powershell
python HuffmanGUI.py
```
