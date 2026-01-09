import secrets

from flask import app

# Generate a secure random key
secret_key = secrets.token_hex(32)  # 64 characters (256 bits)
print(f"Generated Secret Key: {secret_key}")

# Use it in your app
app.secret_key = secret_key