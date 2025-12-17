#!/usr/bin/env python3
"""
Test runner for the RAG Chatbot backend tests
"""
import subprocess
import sys
import os


def run_tests():
    """Run all tests with pytest"""
    try:
        # Change to the backend directory
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(backend_dir)

        print("Running backend tests...")

        # Run pytest with verbose output
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            "tests/",
            "-v",
            "--tb=short",
            "-x"  # Stop after first failure
        ], check=True)

        print("All tests passed! ✅")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Tests failed with return code {e.returncode} ❌")
        return False
    except Exception as e:
        print(f"Error running tests: {e}")
        return False


def run_tests_with_coverage():
    """Run tests with coverage report"""
    try:
        # Change to the backend directory
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(backend_dir)

        print("Running tests with coverage...")

        # Install coverage if not already installed
        subprocess.run([sys.executable, "-m", "pip", "install", "pytest-cov"], check=False)

        # Run pytest with coverage
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            "tests/",
            "--cov=src/",
            "--cov-report=html",
            "--cov-report=term-missing",
            "-v"
        ], check=True)

        print("Coverage report generated in htmlcov/ directory")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Tests failed with return code {e.returncode}")
        return False
    except Exception as e:
        print(f"Error running tests with coverage: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test runner for RAG Chatbot backend")
    parser.add_argument("--coverage", action="store_true", help="Run tests with coverage report")
    args = parser.parse_args()

    if args.coverage:
        success = run_tests_with_coverage()
    else:
        success = run_tests()

    sys.exit(0 if success else 1)