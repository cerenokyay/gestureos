import argparse
from gestureos.app import run

def main():
    p = argparse.ArgumentParser(prog="gestureos")
    p.add_argument("--config", default="config/default.json")
    args = p.parse_args()
    run(args.config)

if __name__ == "__main__":
    main()