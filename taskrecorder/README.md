# Task recorder

A small program that records how a person does one task on their computer, on their computer,
so that the record can later teach a tool or an agent to do that task for them.

It is a local web page served by a Python program. Nothing is recorded until the user names a
task and presses Start; a red light shows while it records; everything it writes lands in one
folder on the same machine, which belongs to the practice that uses it. The person who made the
recording can redact any entry, and the maker of the recorder never receives a copy.

## What it records

* The steps, notes and decisions the user types while working. Enter adds a step.
* Which program and window is in front, and when that changes. Window titles only, sampled every
  two seconds while a task records, and only when the box "Record which window is in front" is
  ticked at start. It can be switched off for good with `--no-focus`.
* A screenshot, only when the user clicks **Capture**.
* Files the user attaches.

It does not record keystrokes, the clipboard, audio, or the screen continuously. While paused,
it records nothing at all, including window titles. Removing an entry with its × deletes its
text, and its screenshot or file, from disk.

When the task ends, the recorder writes `procedure.md`: a numbered draft procedure with the
decisions and notes in place, a table of the programs used and the time spent in each, the
outcome, how tedious the task was on a 1 to 5 scale, and what the user wished a tool had done.
`python -m taskrecorder --export procedures.md` joins every finished procedure into one document.

## Install on a practice computer

1. Install Python 3.10 or later (python.org, or the Microsoft Store on Windows). Nothing else is
   needed: the recorder uses only Python's standard library.
2. Copy the `taskrecorder` folder onto the computer, for example to `C:\Tools\taskrecorder` or
   `~/Tools/taskrecorder`.
3. From the folder that contains it, run

       python -m taskrecorder

   The browser opens on a link that carries a key for this run (`http://127.0.0.1:8766/?t=...`,
   also printed in the terminal). The page does not open without it, so another person or
   program on the same computer cannot read the recordings through the recorder. Recordings go
   to `~/Task Recordings` unless `--dir` says otherwise.

On macOS, recording the window in front and taking screenshots asks for Accessibility and
Screen Recording permission the first time; the system shows the prompt. On Windows both work
without prompts. On Linux the window probe needs `xdotool` and X11.

## Where the data goes

```
~/Task Recordings/
  2026-11-05-review-search-terms/
    task.json        name, goal, owner, tags, start and end, outcome, tedium score, wish
    events.jsonl     every entry in order: id, timestamp, kind, text (+ app/title, file)
    captures/        screenshots taken on request
    files/           attached files
    procedure.md     the draft procedure
```

Keep the folder on an encrypted disk (FileVault or BitLocker) and inside the practice's backup.
The recorder does not encrypt files itself in this version.

## For the engineer

`taskrecorder.store.Store` owns the folder and the rules (one task at a time, redaction, the
procedure rendering). `taskrecorder.probe` holds the operating-system probes, each returning None
when it cannot work. `taskrecorder.app` is the HTTP server and the sampling thread; its docstring
lists the endpoints. Tests: `pytest tests/test_recorder.py`.
