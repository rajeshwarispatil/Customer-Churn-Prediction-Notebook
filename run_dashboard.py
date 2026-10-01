import subprocess, sys

subprocess.run([
    sys.executable, "-m", "streamlit", "run", "app.py",
    "--server.headless", "true", "--server.port", "8501"
])