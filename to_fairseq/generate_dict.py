import json

if __name__ == "__main__":
    import sys
    with open(sys.argv[1], 'r') as f:
        encoder = json.load(f)

    for i in range(len(encoder)):
        print(f'{i} 0')
