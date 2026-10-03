"""Script to set up core Modal secrets for the face recognition system."""

import os
import subprocess


def set_modal_secret():
    """Set up the core Modal secret with essential configuration."""
    try:
        # Create secrets using key=value format
        command = [
            "modal",
            "secret",
            "create",
            "core-secrets",
            f"DATABASE_URL={os.environ['DATABASE_URL']}",
            "MODEL_PATH=/root/models",
            "CONFIDENCE_THRESHOLD=0.6",
            "USE_GPU=false",
            "ENABLE_CACHING=false",
        ]

        subprocess.run(
            command,
            capture_output=True,
            check=True,
            encoding="utf-8",
        )
        print("✅ Successfully set core secrets")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to set core secrets: {e.stderr}")
        return False


if __name__ == "__main__":
    print("Setting up core Modal secrets...")
    set_modal_secret()
    print("\n📝 Next steps:")
    print("1. Verify secrets are set: modal secret list")
    print("2. Deploy your application: modal deploy modal_app/modal_app.py")
