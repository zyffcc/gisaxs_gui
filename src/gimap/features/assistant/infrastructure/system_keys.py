"""Optional OS credential storage. Each GIMaP user data folder has its own namespace."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path


class WindowsCredentials:
    def __init__(self, service, api, error):
        self.service = service
        self.api = api
        self.error = error

    def read(self, account):
        try:
            item = self.api.CredRead(f"{self.service}/{account}", self.api.CRED_TYPE_GENERIC)
        except self.error as exc:
            if exc.winerror == 1168:  # ERROR_NOT_FOUND
                return None
            raise OSError(str(exc)) from exc
        return item["CredentialBlob"].decode("utf-16-le")

    def write(self, account, key):
        try:
            self.api.CredWrite({
                "Type": self.api.CRED_TYPE_GENERIC, "TargetName": f"{self.service}/{account}",
                "UserName": account, "CredentialBlob": key, "Persist": self.api.CRED_PERSIST_LOCAL_MACHINE,
            })
        except self.error as exc:
            raise OSError(str(exc)) from exc

    def delete(self, account):
        try:
            self.api.CredDelete(f"{self.service}/{account}", self.api.CRED_TYPE_GENERIC)
        except self.error as exc:
            if exc.winerror == 1168:
                return False
            raise OSError(str(exc)) from exc
        return True


class SystemKeyring:
    def __init__(self, service, backend, error):
        self.service = service
        self.backend = backend
        self.error = error

    def read(self, account):
        try:
            return self.backend.get_password(self.service, account)
        except self.error as exc:
            raise OSError(str(exc)) from exc

    def write(self, account, key):
        try:
            self.backend.set_password(self.service, account, key)
        except self.error as exc:
            raise OSError(str(exc)) from exc

    def delete(self, account):
        if self.read(account) is None:
            return False
        try:
            self.backend.delete_password(self.service, account)
        except self.error as exc:
            raise OSError(str(exc)) from exc
        return True


def system_credentials(data_dir):
    if data_dir is None:
        return None
    folder = os.path.normcase(str(Path(data_dir).resolve()))
    service = "GIMaP/" + hashlib.sha256(os.fsencode(folder)).hexdigest()
    if sys.platform == "win32":
        try:
            from win32ctypes.pywin32 import pywintypes, win32cred
        except ImportError:
            pass
        else:
            return WindowsCredentials(service, win32cred, pywintypes.error)
    try:
        import keyring
        from keyring.errors import KeyringError
    except ImportError:
        return None
    try:
        backend = keyring.get_keyring()
        candidates = getattr(backend, "backends", [backend])
    except (ImportError, OSError, RuntimeError, KeyringError):
        return None
    os_backends = {"keyring.backends." + name for name in ("Windows", "macOS", "libsecret", "SecretService", "kwallet")}
    for candidate in candidates:
        if type(candidate).__module__ in os_backends:
            return SystemKeyring(service, candidate, KeyringError)
    return None
