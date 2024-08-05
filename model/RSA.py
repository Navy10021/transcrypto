from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP

# Generate RSA Key function
key = RSA.generate(2048)
private_key = key
public_key = key.publickey()

def rsa_encrypt(public_key, data):
    cipher = PKCS1_OAEP.new(public_key)
    return cipher.encrypt(data)

def rsa_decrypt(private_key, data):
    cipher = PKCS1_OAEP.new(private_key)
    return cipher.decrypt(data)
