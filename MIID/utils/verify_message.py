from binascii import unhexlify
from substrateinterface import Keypair

def verify_message(file_path: str) -> None:
    """
    Verifies a signed message from a file using the address and signature contained within.
    
    Args:
        file_path (str): Path to the file containing the message, signer address, and signature.
    
    Raises:
        ValueError: If the signature is invalid or the message is not properly formatted.
    
    Example File Format:
        <Bytes>On 2025-01-01 CST My test message</Bytes>
            Signed by: 5Dw7zQR4s2mZ7N9TMSqtc1Vj41E8ervu57Gh4HMVJ19KQ8Qh
            Signature: f5a67c...

    Usage:
        from my_verify_module import verify_message
        
        try:
            verify_message("message_and_signature.txt")
            print("Signature is valid!")
        except ValueError as e:
            print(f"Invalid signature: {e}")
    """
    # 1) Read the file and split the contents
    with open(file_path, "r", encoding="utf-8") as f:
        file_data = f.read()

    file_split = file_data.split("\n\t")

    # 2) Extract the address
    address_line = file_split[1]
    address_prefix = "Signed by: "
    if address_line.startswith(address_prefix):
        address = address_line[len(address_prefix):]
    else:
        address = address_line

    keypair = Keypair(ss58_address=address, ss58_format=42)

    # 3) Extract the message
    message = file_split[0]
    if not message.startswith("<Bytes>") or not message.endswith("</Bytes>"):
        raise ValueError("Message is not properly wrapped in <Bytes>...</Bytes>.")

    # 4) Extract the signature
    signature_line = file_split[2]
    signature_prefix = "Signature: "
    if signature_line.startswith(signature_prefix):
        signature = signature_line[len(signature_prefix):]
    else:
        signature = signature_line

    real_signature = unhexlify(signature.encode())

    # 5) Verify
    if not keypair.verify(data=message, signature=real_signature):
        raise ValueError(f"Invalid signature for address={address}")
    else:
        print(f"Signature verified, signed by {address}")
