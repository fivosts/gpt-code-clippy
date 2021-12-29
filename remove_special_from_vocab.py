import json
import argparse
import sys
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("vocab_json_in")
    parser.add_argument("vocab_json_out")
    args = parser.parse_args()

    if os.path.exists(args.vocab_json_out):
        print(f"output file {args.vocab_json_out} exists! quitting")
        sys.exit(1)

    with open(args.vocab_json_in, 'r') as f:
        encoder = json.load(f)

    new_encoder = {}

    for k, v in sorted(encoder.items(), key=lambda t: t[1]):
        if k.startswith("<|") and k.endswith("|>"):
            print(f"removing {k} at index {v}")
        else:
            new_encoder[k] = len(new_encoder)

    with open(args.vocab_json_out, 'w') as f:
        json.dump(new_encoder, f, ensure_ascii=False)
