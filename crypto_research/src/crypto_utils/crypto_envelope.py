"""
crypto_envelope.py

Hybrid encryption utilities for the ATLC cryptographic security demo.

This file demonstrates how to protect stored ATLC runtime data, such as:
    - traffic runtime logs
    - signal timing reports
    - operator/exported reports
    - sensitive system state snapshots

Cryptographic design:

    1. Generate a random AES-256 session key.
    2. Encrypt the ATLC plaintext data using AES-GCM.
    3. Encrypt, or "wrap", the AES session key using RSA-OAEP.
    4. Store the encrypted data, encrypted AES key, and metadata separately.
    5. Verify integrity by decrypting the ciphertext and comparing SHA-256 hashes.

Why hybrid encryption is used:

    - AES-GCM is fast and suitable for encrypting larger files.
    - RSA-OAEP is suitable for encrypting small values, such as an AES key.
    - Combining them is the standard hybrid-encryption pattern.

"""

from __future__ import annotations

import base64
import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from crypto_research.src.crypto_utils.file_io import (
    read_bytes,
    write_bytes,
    write_json,
)


@dataclass
class HybridEncryptionResult:
    """
    Structured result returned by hybrid_encrypt_file().

    This makes the demo easier to inspect in:
        - notebook output
        - generated summary JSON
        - generated markdown report

    Fields:
        encrypted_data_path:
            Location of encrypted ATLC data.

        encrypted_key_path:
            Location of RSA-encrypted AES session key.

        metadata_path:
            Location of encryption metadata.

        plaintext_sha256:
            SHA-256 hash of original input data.

        decrypted_sha256:
            SHA-256 hash after decrypting the ciphertext.

        encryption_ms / decryption_ms:
            Runtime cost for encryption and decryption.

        integrity_ok:
            True if decrypted data matches original data.
    """

    encrypted_data_path: Path
    encrypted_key_path: Path
    metadata_path: Path
    plaintext_sha256: str
    decrypted_sha256: str
    encryption_ms: float
    decryption_ms: float
    integrity_ok: bool


def sha256_hex(data: bytes) -> str:
    """
    Compute SHA-256 hash in hexadecimal format.

    Purpose:
        SHA-256 is used here as an integrity check in the demo report.

    Important:
        This hash is not used as encryption.
        It only proves whether two byte strings are identical.
    """

    return hashlib.sha256(data).hexdigest()


def generate_rsa_private_key(key_size: int = 2048):
    """
    Generate a new RSA private key.

    Parameters:
        key_size:
            RSA key length in bits. 2048 bits is commonly used for demos
            and still acceptable for many basic security demonstrations.

    Returns:
        RSA private key object.

    Note:
        The public key can be derived from the private key.
    """

    return rsa.generate_private_key(
        public_exponent=65537,
        key_size=key_size,
    )


def serialize_private_key(private_key) -> bytes:
    """
    Convert an RSA private key object into PEM bytes.

    PEM format is human-readable and starts with:
        -----BEGIN PRIVATE KEY-----

    Security warning:
        This demo stores the private key without password encryption.
        Do not do this in production.
    """

    return private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def serialize_public_key(public_key) -> bytes:
    """
    Convert an RSA public key object into PEM bytes.

    The public key can be shared.
    It is used to encrypt the AES session key.
    """

    return public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def aes_gcm_encrypt(
    plaintext: bytes,
    associated_data: bytes | None = None,
) -> Dict[str, bytes]:
    """
    Encrypt plaintext using AES-256-GCM.

    AES-GCM provides:
        - confidentiality: attacker cannot read plaintext
        - integrity: attacker cannot modify ciphertext without detection

    Parameters:
        plaintext:
            Original ATLC data to encrypt.

        associated_data:
            Optional authenticated context.
            It is not encrypted, but it is included in integrity checking.
            If associated_data changes during decryption, decryption fails.

    Returns:
        Dictionary containing:
            aes_key:
                Random AES-256 session key.

            nonce:
                Random 96-bit nonce required by AES-GCM.

            ciphertext:
                Encrypted data plus authentication tag.
    """

    # Generate a fresh random AES-256 key for this file/session.
    aes_key = AESGCM.generate_key(bit_length=256)

    # AES-GCM standard practice uses a 12-byte nonce.
    # The nonce must be unique for the same AES key.
    nonce = os.urandom(12)

    aesgcm = AESGCM(aes_key)

    ciphertext = aesgcm.encrypt(
        nonce=nonce,
        data=plaintext,
        associated_data=associated_data,
    )

    return {
        "aes_key": aes_key,
        "nonce": nonce,
        "ciphertext": ciphertext,
    }


def aes_gcm_decrypt(
    aes_key: bytes,
    nonce: bytes,
    ciphertext: bytes,
    associated_data: bytes | None = None,
) -> bytes:
    """
    Decrypt AES-GCM ciphertext.

    If the ciphertext, nonce, AES key, or associated data is wrong,
    AES-GCM will raise an exception instead of returning corrupted plaintext.

    This property is important for detecting tampering.
    """

    aesgcm = AESGCM(aes_key)

    return aesgcm.decrypt(
        nonce=nonce,
        data=ciphertext,
        associated_data=associated_data,
    )


def rsa_oaep_wrap_key(public_key, aes_key: bytes) -> bytes:
    """
    Encrypt the AES session key using RSA-OAEP.

    This is called "key wrapping" because RSA is not used to encrypt
    the full data file. It only protects the smaller AES key.

    OAEP with SHA-256 is used because it is safer than old raw RSA encryption.
    """

    return public_key.encrypt(
        aes_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )


def rsa_oaep_unwrap_key(private_key, encrypted_aes_key: bytes) -> bytes:
    """
    Decrypt the RSA-OAEP encrypted AES session key.

    The receiver must have the RSA private key to recover the AES key.
    After that, the AES key is used to decrypt the actual data file.
    """

    return private_key.decrypt(
        encrypted_aes_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )


def hybrid_encrypt_file(
    input_path: Path,
    encrypted_data_path: Path,
    encrypted_key_path: Path,
    metadata_path: Path,
    private_key_path: Path,
    public_key_path: Path,
    associated_data: bytes | None = b"ATLC_CRYPTO_RESEARCH_V1",
) -> HybridEncryptionResult:
    """
    Encrypt one ATLC input file using hybrid encryption.

    This is the main high-level function used by the crypto demo.

    Workflow:
        1. Read plaintext ATLC runtime sample.
        2. Compute SHA-256 hash of plaintext.
        3. Generate RSA key pair.
        4. Encrypt plaintext using AES-GCM.
        5. Encrypt AES key using RSA-OAEP.
        6. Write encrypted outputs.
        7. Immediately decrypt once for verification.
        8. Store metadata and return structured result.

    Parameters:
        input_path:
            Path to plaintext ATLC runtime sample.

        encrypted_data_path:
            Output path for AES-GCM ciphertext.

        encrypted_key_path:
            Output path for RSA-OAEP encrypted AES key.

        metadata_path:
            Output path for JSON metadata.

        private_key_path / public_key_path:
            Output paths for demo RSA keys.

        associated_data:
            Authenticated context string for AES-GCM.
            This binds the ciphertext to this ATLC demo context.

    Returns:
        HybridEncryptionResult with paths, hashes, timing, and integrity status.
    """

    plaintext = read_bytes(input_path)
    plaintext_hash = sha256_hex(plaintext)

    # Generate a temporary RSA key pair for the demo.
    # In real deployment, key generation should happen separately.
    private_key = generate_rsa_private_key()
    public_key = private_key.public_key()

    # Save demo keys so the notebook can show what was generated.
    write_bytes(private_key_path, serialize_private_key(private_key))
    write_bytes(public_key_path, serialize_public_key(public_key))

    encrypt_start = time.perf_counter()

    # Encrypt the actual ATLC data using AES-GCM.
    aes_result = aes_gcm_encrypt(
        plaintext=plaintext,
        associated_data=associated_data,
    )

    # Protect the AES session key using RSA-OAEP.
    encrypted_aes_key = rsa_oaep_wrap_key(
        public_key=public_key,
        aes_key=aes_result["aes_key"],
    )

    encryption_ms = (time.perf_counter() - encrypt_start) * 1000.0

    # Store encrypted payload and encrypted AES key separately.
    write_bytes(encrypted_data_path, aes_result["ciphertext"])
    write_bytes(encrypted_key_path, encrypted_aes_key)

    # Metadata does not contain plaintext or the raw AES key.
    # It contains only information needed to decrypt and verify the demo.
    metadata: Dict[str, Any] = {
        "scheme": "Hybrid AES-256-GCM + RSA-OAEP-SHA256",
        "aes_algorithm": "AES-256-GCM",
        "key_wrapping_algorithm": "RSA-OAEP-SHA256",
        "rsa_key_size": 2048,
        "nonce_b64": base64.b64encode(aes_result["nonce"]).decode("utf-8"),
        "associated_data": associated_data.decode("utf-8") if associated_data else None,
        "plaintext_sha256": plaintext_hash,
        "ciphertext_sha256": sha256_hex(aes_result["ciphertext"]),
        "encrypted_key_sha256": sha256_hex(encrypted_aes_key),
        "plaintext_size_bytes": len(plaintext),
        "ciphertext_size_bytes": len(aes_result["ciphertext"]),
        "encrypted_key_size_bytes": len(encrypted_aes_key),
        "encryption_ms": encryption_ms,
    }

    write_json(metadata_path, metadata)

    # Decrypt immediately once to prove that the encryption result is valid.
    decrypt_start = time.perf_counter()

    recovered_aes_key = rsa_oaep_unwrap_key(
        private_key=private_key,
        encrypted_aes_key=encrypted_aes_key,
    )

    recovered_plaintext = aes_gcm_decrypt(
        aes_key=recovered_aes_key,
        nonce=aes_result["nonce"],
        ciphertext=aes_result["ciphertext"],
        associated_data=associated_data,
    )

    decryption_ms = (time.perf_counter() - decrypt_start) * 1000.0

    decrypted_hash = sha256_hex(recovered_plaintext)
    integrity_ok = plaintext_hash == decrypted_hash

    # Update metadata with verification results.
    metadata["decrypted_sha256"] = decrypted_hash
    metadata["decryption_ms"] = decryption_ms
    metadata["integrity_ok"] = integrity_ok
    write_json(metadata_path, metadata)

    return HybridEncryptionResult(
        encrypted_data_path=encrypted_data_path,
        encrypted_key_path=encrypted_key_path,
        metadata_path=metadata_path,
        plaintext_sha256=plaintext_hash,
        decrypted_sha256=decrypted_hash,
        encryption_ms=encryption_ms,
        decryption_ms=decryption_ms,
        integrity_ok=integrity_ok,
    )


def hybrid_decrypt_file(
    encrypted_data_path: Path,
    encrypted_key_path: Path,
    metadata_path: Path,
    private_key_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    """
    Decrypt a file created by hybrid_encrypt_file().

    This function demonstrates the receiver-side workflow:

        1. Read metadata.
        2. Decode AES-GCM nonce.
        3. Load RSA private key.
        4. Decrypt the AES session key using RSA-OAEP.
        5. Decrypt ciphertext using AES-GCM.
        6. Write recovered plaintext.
        7. Verify plaintext hash.

    Returns:
        Dictionary containing output path, decrypted hash, and integrity status.
    """

    import json

    metadata = metadata_path.read_text(encoding="utf-8")
    metadata_obj = json.loads(metadata)

    nonce = base64.b64decode(metadata_obj["nonce_b64"])

    associated_text = metadata_obj.get("associated_data")
    associated_data = associated_text.encode("utf-8") if associated_text else None

    # Load private key from PEM.
    private_key = serialization.load_pem_private_key(
        private_key_path.read_bytes(),
        password=None,
    )

    encrypted_aes_key = encrypted_key_path.read_bytes()
    ciphertext = encrypted_data_path.read_bytes()

    # Recover AES key first.
    aes_key = rsa_oaep_unwrap_key(
        private_key=private_key,
        encrypted_aes_key=encrypted_aes_key,
    )

    # Use recovered AES key to decrypt the actual data.
    plaintext = aes_gcm_decrypt(
        aes_key=aes_key,
        nonce=nonce,
        ciphertext=ciphertext,
        associated_data=associated_data,
    )

    write_bytes(output_path, plaintext)

    decrypted_hash = sha256_hex(plaintext)

    return {
        "output_path": str(output_path),
        "decrypted_sha256": decrypted_hash,
        "integrity_ok": decrypted_hash == metadata_obj["plaintext_sha256"],
    }