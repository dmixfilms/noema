Patch applied: the last block of servidor.py now reads the port from the
NOEMA_PORT environment variable (default 7860):

```python
if __name__ == "__main__":
    import uvicorn
    porta = int(os.environ.get("NOEMA_PORT", "7860"))
    print(f"\n  Noema Dashboard →  http://localhost:{porta}\n")
    uvicorn.run(app, host="127.0.0.1", port=porta, log_level="warning")
```

Reason: on Windows with Hyper-V/WSL, dynamic port exclusion ranges can block
binding to 7860 (WinError 10013). The startup script defaults to port 8878.
