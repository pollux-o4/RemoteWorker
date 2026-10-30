"""ComfyUI Manager/GenRel install hook for the shared abi3 native extension."""
from tools.build_native import build

if __name__ == "__main__":
    print(build())
