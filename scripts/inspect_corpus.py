import pathlib
from collections import Counter

DATA_ROOT = pathlib.Path("data/raw")

def iter_text_files(root: pathlib.Path):
    for path in root.rglob("*.txt"):
        with path.open("r", encoding="utf-8") as f:
            yield path, f.read()

def main():
    total_chars = 0
    vocab = Counter()

    for path, text in iter_text_files(DATA_ROOT):
        print(f"Loaded {path} ({len(text)} chars)")
        total_chars += len(text)
        vocab.update(text)  # character-level for now

    print("\n=== Summary ===")
    print(f"Total characters: {total_chars}")
    print(f"Unique characters: {len(vocab)}")
    print("Top 30 characters:")
    for ch, count in vocab.most_common(30):
        display = ch if ch != "\n" else "\\n"
        print(f"  {repr(display):>6} : {count}")

if __name__ == "__main__":
    main()
