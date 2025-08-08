import os
import subprocess
import platform

def open_image(path):
    if platform.system() == "Darwin":  # macOS
        subprocess.run(["open", path])
    elif platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Linux":
        subprocess.run(["xdg-open", path])
    else:
        raise RuntimeError("Unsupported operating system")

def open_all_pngs(directory):
    for file in os.listdir(directory):
        if file.lower().endswith(".png"):
            full_path = os.path.join(directory, file)
            open_image(full_path)

if __name__ == "__main__":
    open_all_pngs(os.path.dirname(os.path.abspath(__file__)))

