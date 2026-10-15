"""Start the local app without Streamlit's email prompt; stop with Ctrl+C."""
import argparse
import importlib.util
import importlib
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--check", action="store_true", help="Check dependencies without starting the server")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Choose a port between 1024 and 65535.")
    missing = [name for name in ("streamlit", "numpy", "scipy", "rasterio", "skimage") if importlib.util.find_spec(name) is None]
    if missing:
        parser.exit(1, f"Missing packages: {', '.join(missing)}. Install requirements-core.txt.\n")
    if args.check:
        try:
            for name in ("streamlit", "numpy", "scipy", "rasterio", "skimage"):
                importlib.import_module(name)
        except (ImportError, OSError) as exc:
            parser.exit(1, f"A dependency could not be loaded: {exc}\n")
        print("GUI dependencies imported successfully. No server started.")
        return
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", args.port))
        except OSError:
            parser.exit(1, f"Port {args.port} is already in use. Stop the previous instance or run with --port 8502.\n")
    url = f"http://127.0.0.1:{args.port}"
    command = [sys.executable, "-m", "streamlit", "run", str(ROOT / "app/app.py"),
               "--server.address", "127.0.0.1", "--server.port", str(args.port),
               "--server.headless", "true", "--browser.gatherUsageStats", "false"]
    server = subprocess.Popen(command, cwd=ROOT)
    try:
        print(f"Starting Geo Enhance at {url}. Press Ctrl+C here to stop.", flush=True)
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError("The server exited during startup. See the error above.")
            try:
                with urllib.request.urlopen(url + "/_stcore/health", timeout=1) as response:
                    ready = response.status == 200
                if ready:
                    break
            except OSError:
                pass
            time.sleep(.25)
        else:
            raise RuntimeError("Startup timed out. Check the errors above and your local firewall settings.")
        print(f"Ready: {url}", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        code = server.wait()
        if code:
            raise RuntimeError(f"Server exited with code {code}. See the error above.")
    except KeyboardInterrupt:
        print("\nStopping Geo Enhance…", flush=True)
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
