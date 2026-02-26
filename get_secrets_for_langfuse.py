import secrets
import base64

print("=== КЛЮЧИ ДЛЯ LANGFUSE ===")
print("ENCRYPTION_KEY=", secrets.token_hex(32))
print("NEXTAUTH_SECRET=", base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8').rstrip('='))
print("SALT=", base64.urlsafe_b64encode(secrets.token_bytes(24)).decode('utf-8').rstrip('='))
print("REDIS_AUTH=", base64.urlsafe_b64encode(secrets.token_bytes(24)).decode('utf-8').rstrip('='))
