from pathlib import Path
import joblib
import numpy as np

SOURCE = Path("models/dest_state.joblib")
OUTPUT = Path("models")

print("Loading original destination state...")
state = joblib.load(SOURCE)

n = len(state)
print(f"Found {n:,} accounts")

print("Sorting account IDs...")
keys = sorted(state.keys())

if any(len(k.encode("ascii")) > 11 for k in keys):
    raise ValueError("An account ID exceeds 11 ASCII bytes")

print("Creating compact NumPy arrays...")

ids = np.asarray(keys, dtype="S11")

counts = np.fromiter(
    (state[k][0] for k in keys),
    dtype=np.uint32,
    count=n
)

sums = np.fromiter(
    (state[k][1] for k in keys),
    dtype=np.float64,
    count=n
)

if any(keys[i] >= keys[i + 1] for i in range(n - 1)):
    raise ValueError("Account IDs are not unique and sorted")

if not np.all(np.isfinite(sums)):
    raise ValueError("Invalid amount sum detected")

print("Saving NumPy files...")

np.save(OUTPUT / "dest_state_ids.npy", ids)
np.save(OUTPUT / "dest_state_counts.npy", counts)
np.save(OUTPUT / "dest_state_sums.npy", sums)

print("\nConversion successful!")
print("Accounts:", f"{n:,}")
print("ID dtype:", ids.dtype)
print("Count dtype:", counts.dtype)
print("Sum dtype:", sums.dtype)

for filename in [
    "dest_state_ids.npy",
    "dest_state_counts.npy",
    "dest_state_sums.npy",
]:
    path = OUTPUT / filename
    print(f"{filename}: {path.stat().st_size / (1024**2):.2f} MB")

print("\nOriginal joblib file has NOT been modified.")
