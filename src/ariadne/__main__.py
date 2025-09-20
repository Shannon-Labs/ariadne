"""
Ariadne command-line interface.

Usage:
    python -m ariadne calibrate [options]
"""

import sys
from .calibration import run_calibration, save_calibration


def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python -m ariadne <command>")
        print("Commands:")
        print("  calibrate  - Calibrate router for current hardware")
        sys.exit(1)

    command = sys.argv[1]

    if command == "calibrate":
        # Run calibration with default settings
        print("🚀 Starting Ariadne calibration...")
        summary = run_calibration(shots=128, repetitions=3, verbose=True)

        # Save to default location
        output_path = save_calibration(summary)

        print(f"\n✅ Calibration complete! Saved to: {output_path}")
        print("\nYour router will now use these calibrated values automatically.")

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()