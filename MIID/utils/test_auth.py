import os
import bittensor as bt
from MIID.utils.sign_message import sign_message
from MIID.utils.verify_message import verify_message

def run_standalone_test():
    """
    Standalone script to test signing and verifying messages.
    Usage: python3 MIID/utils/test_auth.py
    """
    # 1. Setup: Load your wallet
    # 'default' is the wallet name, 'default' is the hotkey name. Change as needed.
    # You must have a created wallet on your machine (~/.bittensor/wallets/).
    # If no default wallet exists, you might need to specify name/hotkey arguments.
    try:
        # Tries to load the default config wallet, or falls back to 'default'/'default'
        wallet = bt.wallet() 
        print(f"Loaded wallet: {wallet.name}, Hotkey: {wallet.hotkey_str}")
    except Exception as e:
        print(f"Error: Could not load wallet. Make sure you have a bittensor wallet configured.\nDetails: {e}")
        return

    message_text = "This is a test verification message"
    output_filename = "test_auth_signature.txt"

    print(f"--- 1. Signing Message: '{message_text}' ---")
    
    # 2. Sign the message
    # This will:
    #   - Create a timestamped string <Bytes>On 2024-xx-xx ... message</Bytes>
    #   - Sign it with your wallet.hotkey
    #   - Write the result to 'test_auth_signature.txt'
    try:
        signed_content = sign_message(wallet, message_text, output_file=output_filename)
        
        print("\nGenerated Signed Content:")
        print(signed_content)

        print(f"\n--- 2. Verifying File: '{output_filename}' ---")

        # 3. Verify the message
        # This reads the file back and checks if the signature matches the address
        verify_message(output_filename)
        print("\n[SUCCESS] Verification passed! The signature is valid.")
        
    except ValueError as e:
        print(f"\n[FAILURE] Verification failed: {e}")
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
    finally:
        # Cleanup
        if os.path.exists(output_filename):
            os.remove(output_filename)
            print(f"\n(Cleaned up {output_filename})")

if __name__ == "__main__":
    run_standalone_test()
