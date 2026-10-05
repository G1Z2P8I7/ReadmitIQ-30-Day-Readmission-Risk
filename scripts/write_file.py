import sys, base64, pathlib

if len(sys.argv) < 3:
    print('Usage: write_file.py <target_path> <base64_content>')
    sys.exit(1)

target = pathlib.Path(sys.argv[1])
target.parent.mkdir(parents=True, exist_ok=True)
data = base64.b64decode(sys.argv[2])
target.write_bytes(data)
print(f'Wrote {len(data)} bytes to {target}')
