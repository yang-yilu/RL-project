"""
Quick test script to run training with diagnostics and capture output.
"""
import sys
import subprocess
import time

# Modify the MAX_GLOBAL_STEPS in the main script temporarily
print("Starting diagnostic test run (50k steps)...")
print("=" * 80)

# Run the training script
process = subprocess.Popen(
    [sys.executable, "a3c_mario6.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

# Collect output for a reasonable time or until process completes
output_lines = []
diagnostic_lines = []
episode_lines = []
max_wait_time = 300  # 5 minutes max
start_time = time.time()

print("Collecting output (will stop after 5 minutes or when training completes)...")
print("-" * 80)

try:
    for line in process.stdout:
        output_lines.append(line)
        print(line, end='')  # Print in real-time
        
        # Categorize lines
        if "Update" in line and "Rollout rewards" in line:
            diagnostic_lines.append(line.strip())
        elif "Ep " in line and "Global steps" in line:
            episode_lines.append(line.strip())
        
        # Stop if we've collected enough or time limit reached
        if len(diagnostic_lines) >= 20 or (time.time() - start_time) > max_wait_time:
            print("\n" + "=" * 80)
            print("Stopping collection (enough diagnostics collected or time limit reached)")
            process.terminate()
            break
            
except KeyboardInterrupt:
    process.terminate()
    print("\nInterrupted by user")

process.wait()

print("\n" + "=" * 80)
print("DIAGNOSTIC SUMMARY")
print("=" * 80)
print(f"\nTotal diagnostic lines collected: {len(diagnostic_lines)}")
print(f"Total episode lines collected: {len(episode_lines)}")

if diagnostic_lines:
    print("\nSample Diagnostic Outputs:")
    print("-" * 80)
    for line in diagnostic_lines[:10]:
        print(line)
    
    # Analyze reward ranges
    import re
    rewards = []
    grad_norms = []
    for line in diagnostic_lines:
        # Extract mean reward
        mean_match = re.search(r'mean=([-\d.]+)', line)
        if mean_match:
            rewards.append(float(mean_match.group(1)))
        # Extract grad norm
        grad_match = re.search(r'Grad norm: ([\d.]+)', line)
        if grad_match:
            grad_norms.append(float(grad_match.group(1)))
    
    if rewards:
        print(f"\nReward Statistics:")
        print(f"  Mean reward range: {min(rewards):.3f} to {max(rewards):.3f}")
        print(f"  Average mean reward: {sum(rewards)/len(rewards):.3f}")
    
    if grad_norms:
        print(f"\nGradient Norm Statistics:")
        print(f"  Range: {min(grad_norms):.4f} to {max(grad_norms):.4f}")
        print(f"  Average: {sum(grad_norms)/len(grad_norms):.4f}")
    
    # Check for NaN/Inf
    has_nan = any("[NaN DETECTED!]" in line for line in diagnostic_lines)
    has_inf = any("[Inf DETECTED!]" in line for line in diagnostic_lines)
    print(f"\nNaN detected: {has_nan}")
    print(f"Inf detected: {has_inf}")

if episode_lines:
    print(f"\nSample Episode Outputs (last 5):")
    print("-" * 80)
    for line in episode_lines[-5:]:
        print(line)

print("\n" + "=" * 80)

