# HMove: An Intelligent Refactoring Plugin

## Plugin Overview

The HMove Intelligent Refactoring Plugin is an extension for performing intelligent code refactoring in Visual Studio Code. It provides a set of prediction and refactoring tools to help developers improve code quality and readability. Our prototype implementation adopts a two-layer C/S architecture: the client layer for interaction with users and the server layer for refactoring opportunity detection. It is worth noting that in the server layer, we have deployed a large model for automated verification of the legitimacy of refactoring candidates.

## Installation

1. Download the .vsix file from our package.
2. Open Visual Studio Code (VS Code).
3. Go to the Extensions view (Ctrl+Shift+X).
4. Click the menu button (three dots) at the top right of the Extensions view.
5. Select "Install from VSIX."
6. Navigate to and open .vsix extension file using the file browser.

## Usage

1. Open your test project.
2. Open the Java project that you want to refactor in the editor.
3. Click the HMove icon in the bottom-right corner to open the plugin interface.
4. In the plugin interface, select the project and the refactoring operation:
   - Choose the Java source project to refactor.
   - detection: Detect the methods requiring refactoring along with their corresponding source and target classes, forming a table of refactoring recommendations.
   - refactoring: Execute the code refactoring.
5. Follow the prompts to complete the refactoring.

## Plugin Entry Point

You can find the HMove icon in the bottom-right corner of VS Code. Click it to open the plugin interface.