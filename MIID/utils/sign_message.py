from datetime import datetime
import bittensor


def sign_message(wallet_name: str, message_text: str, output_file: str = "message_and_signature.txt"):
    """
    Signs a message using the specified wallet and writes it to a file.

    Args:
        wallet_name (str): Name of the wallet to load (coldkey).
        message_text (str): The message you want to sign.
        output_file (str, optional): Filename to save message and signature. Defaults to "message_and_signature.txt".

    Returns:
        str: The combined file contents (message + signature info).
    """
    # Create a wallet object
    wallet = bittensor.wallet(name=wallet_name)
    keypair = wallet.coldkey

    # Generate a timestamped message
    timestamp = datetime.now()
    timezone_name = timestamp.astimezone().tzname()
    signed_message = f"<Bytes>On {timestamp} {timezone_name} {message_text}</Bytes>"

    # Sign the message
    signature = keypair.sign(data=signed_message)

    # Construct the output
    file_contents = (
        f"{signed_message}\n"
        f"\tSigned by: {keypair.ss58_address}\n"
        f"\tSignature: {signature.hex()}"
    )

    # Print to console
    print(file_contents)

    # Write to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(file_contents)

    print(f"Signature generated and saved to {output_file}")
    return file_contents
