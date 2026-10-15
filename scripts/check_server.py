"""Start a temporary local server, verify HTTP readiness, and always stop it."""
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', str(ROOT / 'app/app.py'),
                                    '--server.address', '127.0.0.1', '--server.port', str(port),
                                    '--server.headless', 'true', '--browser.gatherUsageStats', 'false'],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError('Server exited during startup.')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/_stcore/health', timeout=1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                time.sleep(.25)
            else:
                raise RuntimeError('Server readiness timed out.')
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/', timeout=5) as response:
                if response.status != 200 or b'<html' not in response.read().lower():
                    raise RuntimeError('The app HTML was not served.')
            print('Local server health and HTML checks passed.')
        except Exception:
            log.seek(0)
            print(log.read().decode('utf-8', errors='replace'), file=sys.stderr)
            raise
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            print('Temporary test server stopped.')


if __name__ == '__main__':
    main()
