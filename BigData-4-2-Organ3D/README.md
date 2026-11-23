To test the tool immediately, visit https://tarjeim.github.io/organ3d/

## What you need

- **Node.js** (which includes **npm**), a current LTS version.
- A modern browser with WebGL (Chrome, Edge, Firefox or Safari).

### Check whether Node is already installed

Open a terminal (Terminal on macOS and Linux, PowerShell or Command Prompt on Windows) and run:

```
node -v
npm -v
```

If both print a version number, skip to "Set up the project". If you get "command not found" or "not recognized", install Node first.

### Installing Node.js

Download the **LTS** installer from https://nodejs.org and run it with the default options. Then close and reopen your terminal and run `node -v` again to confirm.

## Set up the project

The project has four files and they must all be in the same folder:

```
index.html
main.js
pbd.js
package.json
```

Open a terminal in that folder (`cd path/to/the/folder`), then install the dependencies:

```
npm install three lil-gui
npm install --save-dev vite
```

This creates a `node_modules` folder and a `package-lock.json`. You only need to do this once.

## Run it

```
npm run dev
```

Vite prints a local address, usually `http://localhost:5173`. Open it in your browser. 
Press `Ctrl+C` in the terminal to stop the server.
