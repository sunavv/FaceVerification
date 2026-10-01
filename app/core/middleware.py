"""Security Headers and Request Processing Middleware."""

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Receive, Scope, Send


class SecurityHeadersMiddleware:
    """
    ASGI middleware enforcing robust security headers across all responses.

    Protects against:
    - Cross-Site Scripting (CSP, X-XSS-Protection)
    - Clickjacking (X-Frame-Options, CSP frame-ancestors)
    - MIME-type sniffing (X-Content-Type-Options: nosniff)
    - Protocol downgrade attacks (Strict-Transport-Security HSTS)
    - Referrer leakage (Referrer-Policy: strict-origin-when-cross-origin)
    - Unintended device access (Permissions-Policy: camera=(self), microphone=(), etc.)
    - Cross-origin isolation issues (COOP, CORP)
    - Server fingerprinting (stripping Server header)
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_security_headers(message):
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)

                # Content-Security-Policy (CSP)
                headers["Content-Security-Policy"] = (
                    "default-src 'self'; "
                    "script-src 'self'; "
                    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                    "font-src 'self' https://fonts.gstatic.com data:; "
                    "img-src 'self' data: blob: http: https:; "
                    "media-src 'self' blob:; "
                    "connect-src 'self' http: https: ws: wss:; "
                    "frame-ancestors 'none'; "
                    "base-uri 'self'; "
                    "form-action 'self'"
                )

                # HTTP Strict Transport Security (HSTS) with preload
                headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"

                # Prevent MIME sniffing
                headers["X-Content-Type-Options"] = "nosniff"

                # Frame options & modern XSS protection (disable buggy legacy auditor in favor of CSP)
                headers["X-Frame-Options"] = "DENY"
                headers["X-XSS-Protection"] = "0"

                # Referrer Policy
                headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

                # Permissions Policy: camera allowed for self (webcam biometric capture), block unused peripherals
                headers["Permissions-Policy"] = (
                    "camera=(self), microphone=(), geolocation=(), payment=(), usb=()"
                )

                # Cross-Origin Policies
                headers["Cross-Origin-Opener-Policy"] = "same-origin"
                headers["Cross-Origin-Embedder-Policy"] = "credentialless"
                headers["Cross-Origin-Resource-Policy"] = "same-origin"

                # Mask or remove Server header to prevent fingerprinting
                if "server" in headers:
                    del headers["server"]

            await send(message)

        await self.app(scope, receive, send_with_security_headers)
