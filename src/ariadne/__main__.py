import argparse

def main():
    """Main command-line interface for Ariadne."""
    parser = argparse.ArgumentParser(
        prog="ariadne",
        description="Ariadne: The Intelligent Quantum Router."
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0"
    )
    # Future sub-commands for simulate, route, benchmark, etc. can be added here.
    args = parser.parse_args()

    print("Welcome to Ariadne. Use --help for options.")

if __name__ == "__main__":
    main()
