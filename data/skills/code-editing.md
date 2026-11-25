# Create and save a user-requested text/code file

VS Code accessibility prerequisite: if only window buttons and an opaque DocumentControl are visible, do not type into that container or use untargeted typing to bypass validation. Open Accessibility Help (Alt+F1 on Windows) once when keyboard input is permitted. Inspect help and activate an observed enable-screen-reader control if offered, then return to the editor and reobserve. Do not blindly toggle a mode repeatedly. If controls remain unavailable, report the accessibility blocker. Official reference: https://code.visualstudio.com/docs/configure/accessibility/accessibility

Use the user's filename, destination, language, and requested content as parameters.
Do not substitute a memorized example or execute code unless asked.

1. Inspect the foreground editor. Reuse the requested app if already open. For an unspecified editor, Notepad is suitable for a plain text/code file.
2. Create a NEW document (Ctrl+N in VS Code/Notepad when keyboard is allowed). Never replace the contents of an unrelated existing document. If an unsaved-work dialog appears, preserve that work and report the conflict.
3. Focus the observed editor field and enter the requested content. In VS Code distinguish the editor from the terminal, search box, and command palette. Inspect the resulting content, indentation, and quotes; an input receipt alone does not prove correct code.
4. Save to the full destination path. Use the provided local Desktop directory when requested. Ctrl+Shift+S opens Save As in these editors when shortcuts are allowed. Wait for and inspect the dialog before typing its filename.
5. Preserve the exact extension: in Notepad choose All files when available so a requested .py or .js file is not silently saved as .txt. Do not approve overwriting an existing file unless the user requested that replacement.
6. Confirm the saved file and contents. If verification fails, inspect the actual error and correct that stage only. Do not restart the whole workflow or keep repeating the same shortcut.

Interaction constraints override these reference shortcuts. If only launch requires mouse, use mouse until the requested app is visibly foreground, then normal typing/shortcuts are allowed. For mouse-based saving, click observed File/Save controls; text entry into the filename is allowed unless the whole task forbids keyboard input. If required controls cannot be observed, explain the blocker rather than inventing coordinates or claiming success.
