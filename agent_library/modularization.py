import os
import time
from cursor_agent import SyncClient

# ---------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------

REPO_URL = "https://github.com/Sowmya-0130/Unified-SingleDB"
BRANCH = "main"

PROMPT = """
You are a senior software architect.

Your task is to modularize this repository.

Requirements:

1. Analyze the architecture first.
2. Identify:
   - God classes
   - Large files (>500 LOC)
   - Circular dependencies
   - Duplicate code
   - Dead code
   - Utility functions that should be extracted

3. Refactor incrementally.

4. Preserve functionality.

5. Do NOT introduce breaking API changes.

6. Run all tests after each major refactor.

7. Fix any failing tests.

8. Keep commits small and meaningful.

9. When complete:
   - Summarize all architectural improvements.
   - Push changes to a new branch.
   - Automatically create a Pull Request.
"""

# ---------------------------------------------------
# MAIN
# ---------------------------------------------------

def main():

    api_key = os.environ.get("CURSOR_API_KEY")

    if not api_key:
        raise RuntimeError(
            "CURSOR_API_KEY environment variable is not set."
        )

    with SyncClient(api_key) as client:

        print("Creating Cloud Agent...")

        agent = client.new_agent(
            repo=REPO_URL,
            ref=BRANCH
        )

        print("Launching task...\n")

        response = agent.create(
            PROMPT,
            target={
                "autoCreatePr": True
            }
        )

        print("Agent launched.")
        print(response)

        print("\nWaiting for completion...\n")

        while True:

            status = agent.refresh()

            state = status.get("status")

            print("Current status:", state)

            if state in (
                "completed",
                "failed",
                "cancelled",
                "error",
            ):
                break

            time.sleep(20)

        print("\nFinal Status")
        print("---------------------")
        print(status)


if __name__ == "__main__":
    main()