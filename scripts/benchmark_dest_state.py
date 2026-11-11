from pathlib import Path
import resource
import time

from src.features import CustomerStateCache

state_file = Path("models/dest_state.joblib")

print("Loading 2.63 million accounts using memory mapping...")
start = time.perf_counter()

cache = CustomerStateCache(state_file=state_file)

load_time = time.perf_counter() - start
memory_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

print(f"Load time: {load_time:.3f} seconds")
print(f"Peak process memory: {memory_mb:.2f} MB")

# Verify a known historical account.
account = "C1000004082"
expected_count = 6
expected_sum = 2259324.39

count, total, mean = cache.get_destination_history(account)

print("\nHistorical state check:")
print("Count:", count)
print("Sum:", total)
print("Mean:", mean)

assert count == expected_count
assert abs(total - expected_sum) < 0.01

# Benchmark 10,000 lookups.
iterations = 10000
start = time.perf_counter()

for _ in range(iterations):
    cache.get_destination_history(account)

elapsed = time.perf_counter() - start

print("\nLookup benchmark:")
print("Lookups:", iterations)
print(f"Total time: {elapsed:.4f} seconds")
print(f"Average lookup: {(elapsed / iterations) * 1000:.4f} ms")

# Verify update logic, then restore the original values.
idx = cache._find_index(account)
original_count = int(cache._counts[idx])
original_sum = float(cache._sums[idx])

cache.update_destination(account, 100.0)

assert int(cache._counts[idx]) == original_count + 1
assert abs(float(cache._sums[idx]) - (original_sum + 100.0)) < 0.01

cache._counts[idx] = original_count
cache._sums[idx] = original_sum
cache._counts.flush()
cache._sums.flush()

print("\nExisting account update: PASSED")
print("Original account values restored.")
print("\nBenchmark completed successfully.")
