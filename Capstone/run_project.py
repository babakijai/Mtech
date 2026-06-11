from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def run_command(command: list[str], description: str) -> None:
    print(f"\n==> {description}")
    print(" ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def run_pipeline() -> None:
    run_command(
        [sys.executable, "-m", "src.models.train_isolation_forest"],
        "Training Isolation Forest model",
    )
    run_command(
        [sys.executable, "-m", "src.models.predict"],
        "Scoring billing records",
    )
    run_command(
        [sys.executable, "-m", "src.rag.create_embeddings"],
        "Generating RAG anomaly reports",
    )


def start_api(host: str, port: int) -> None:
    print(f"\nFastAPI docs: http://{host}:{port}/docs")
    run_command(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "src.api.app:app",
            "--host",
            host,
            "--port",
            str(port),
        ],
        "Starting FastAPI server",
    )


def wait_for_url(url: str, timeout_seconds: int = 30) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def start_ui(port: int) -> None:
    url = f"http://localhost:{port}"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "src/ui/streamlit_app.py",
        "--server.headless",
        "true",
        "--server.port",
        str(port),
        "--server.fileWatcherType",
        "none",
    ]

    print("\n==> Starting Streamlit dashboard")
    print(" ".join(command))
    process = subprocess.Popen(command, cwd=PROJECT_ROOT)

    if wait_for_url(url):
        print(f"\nOpening dashboard: {url}")
        webbrowser.open(url)
    else:
        print(f"\nDashboard is starting. Open manually if needed: {url}")

    try:
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait()


def start_project(port: int) -> None:
    start_ui(port)


def create_review_sample() -> None:
    run_command(
        [sys.executable, "-m", "src.evaluation.validation", "--mode", "sample"],
        "Creating anomaly review label sample",
    )


def evaluate_labels() -> None:
    run_command(
        [sys.executable, "-m", "src.evaluation.validation", "--mode", "evaluate"],
        "Evaluating reviewer labels",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Hospital Billing AI Investigation Assistant."
    )
    parser.add_argument(
        "target",
        choices=["api", "ui", "pipeline", "review-sample", "evaluate"],
        nargs="?",
        default="ui",
        help="What to run. Default: ui.",
    )
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="Start the API or UI without retraining, rescoring, or regenerating RAG reports.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for the FastAPI server. Default: 127.0.0.1.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port for the FastAPI server. Default: 8001.",
    )
    parser.add_argument(
        "--ui-port",
        type=int,
        default=8501,
        help="Port for the Streamlit dashboard. Default: 8501.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.target == "pipeline":
        run_pipeline()
        return

    if args.target == "review-sample":
        create_review_sample()
        return

    if args.target == "evaluate":
        evaluate_labels()
        return

    if not args.skip_pipeline:
        run_pipeline()

    if args.target == "api":
        start_api(args.host, args.port)
    elif args.target == "ui":
        start_project(args.ui_port)


if __name__ == "__main__":
    main()
