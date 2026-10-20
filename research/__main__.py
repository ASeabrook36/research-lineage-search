import argparse
import json
import logging


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("azure").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="Local EUV research discovery")
    parser.add_argument("command", choices=["probe", "collect", "ingest", "evaluate"])
    command = parser.parse_args().command
    if command == "probe":
        from research.db import probe
        result = probe()
    elif command == "collect":
        from research.collect import collect
        result = collect()
    elif command == "ingest":
        from research.ingest import ingest
        result = ingest()
    else:
        from research.evaluate import evaluate
        result = evaluate()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
