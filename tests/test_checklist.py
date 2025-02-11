# post-pilot/tests/test_checklist.py

import os
import sys

# Get the absolute paths to the directories
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

def verify_dashboard_setup():
    """
    Run these checks manually and verify each returns True
    """
    checks = {
        "directory_structure": check_directory_structure(),
        "files_exist": check_required_files(),
        "dependencies_installed": check_dependencies()
    }
    return all(checks.values()), checks

def check_directory_structure():
    required_dirs = [
        os.path.join(PROJECT_ROOT, "components", "dashboard"),
        os.path.join(PROJECT_ROOT, "components", "dashboard", "static")
    ]
    results = {dir: os.path.exists(dir) for dir in required_dirs}
    print("\nChecking directory structure:")
    for dir, exists in results.items():
        print(f"{dir}: {'✓' if exists else '✗'}")
    return all(results.values())

def check_required_files():
    required_files = [
        os.path.join(PROJECT_ROOT, "components", "dashboard", "dashboard_tab.py"),
        os.path.join(PROJECT_ROOT, "components", "dashboard", "static", "dashboard.html"),
        os.path.join(PROJECT_ROOT, "components", "dashboard", "static", "dashboard.js"),
        os.path.join(PROJECT_ROOT, "components", "dashboard", "__init__.py")
    ]
    results = {file: os.path.exists(file) for file in required_files}
    print("\nChecking required files:")
    for file, exists in results.items():
        print(f"{file}: {'✓' if exists else '✗'}")
    return all(results.values())

def check_dependencies():
    try:
        import PyQt5.QtWebEngineWidgets
        print("\nChecking dependencies:")
        print("PyQt5.QtWebEngineWidgets: ✓")
        return True
    except ImportError:
        print("\nChecking dependencies:")
        print("PyQt5.QtWebEngineWidgets: ✗ (not installed)")
        return False

if __name__ == "__main__":
    success, results = verify_dashboard_setup()
    print("\nOverall check results:")
    for check, passed in results.items():
        print(f"{check}: {'✓' if passed else '✗'}")
    print(f"\nFinal result: {'✓ All checks passed' if success else '✗ Some checks failed'}")