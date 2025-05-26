"""Script to set up core Modal secrets for the face recognition system."""

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
            "DATABASE_URL=postgresql://neondb_owner:npg_pKZdqik8DP9w@ep-hidden-bush-a8h3pypg-pooler.eastus2.azure.neon.tech/neondb?sslmode=require",
            "MODEL_PATH=/root/models",
            "CONFIDENCE_THRESHOLD=0.6",
            "USE_GPU=false",
            "ENABLE_CACHING=false",
        ]

        result = subprocess.run(
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
