import os

def get_roots() -> tuple[str, str]:
    """
    Calculates and returns the project root and package root directories.

    The project root is the HAR-25/ directory.
    The package root is HAR-25/src/har/.

    Returns:
        tuple[str, str]: A tuple containing (project_root, package_root).
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))

    package_root = current_dir

    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    return project_root, package_root