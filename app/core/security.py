from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import jwt
from passlib.context import CryptContext
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import secrets
import structlog
from redis import Redis
from ratelimit import RateLimitException, RateLimitDecorator
import hashlib
import magic
import os
from PIL import Image
import io

# Initialize logger
logger = structlog.get_logger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT configuration
JWT_SECRET = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "7"))

# File security
MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", "10485760"))  # 10MB
ALLOWED_IMAGE_TYPES = os.getenv("ALLOWED_IMAGE_TYPES", "image/jpeg,image/png").split(
    ","
)
MIN_IMAGE_SIZE = (160, 160)

# Redis configuration for rate limiting and token blacklist
redis_client = Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    db=int(os.getenv("REDIS_DB", "0")),
    password=os.getenv("REDIS_PASSWORD"),
    ssl=os.getenv("REDIS_SSL", "false").lower() == "true",
)


class SecurityError(Exception):
    """Base class for security-related errors."""

    pass


class InvalidTokenError(SecurityError):
    """Raised when a token is invalid."""

    pass


class RateLimitExceededError(SecurityError):
    """Raised when rate limit is exceeded."""

    pass


class FileValidationError(SecurityError):
    """Raised when file validation fails."""

    pass


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)


def create_access_token(
    data: Dict[str, Any], expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT access token.

    Args:
        data: Data to encode in the token
        expires_delta: Token expiration time

    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update(
        {"exp": expire, "iat": datetime.utcnow(), "jti": secrets.token_urlsafe(32)}
    )

    try:
        encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
        return encoded_jwt
    except Exception as e:
        logger.error("Token creation failed", error=str(e))
        raise SecurityError("Could not create access token")


def create_refresh_token(user_id: str) -> str:
    """Create a refresh token."""
    expires_delta = timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    return create_access_token(
        data={"sub": user_id, "type": "refresh"}, expires_delta=expires_delta
    )


def decode_token(token: str) -> Dict[str, Any]:
    """
    Decode and validate a JWT token.

    Args:
        token: JWT token to decode

    Returns:
        Decoded token data

    Raises:
        InvalidTokenError: If token is invalid or expired
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])

        # Check if token is blacklisted
        if redis_client.exists(f"blacklist:{payload['jti']}"):
            raise InvalidTokenError("Token has been revoked")

        return payload

    except jwt.ExpiredSignatureError:
        raise InvalidTokenError("Token has expired")
    except jwt.JWTError as e:
        raise InvalidTokenError(f"Invalid token: {str(e)}")


def blacklist_token(token: str):
    """Add a token to the blacklist."""
    try:
        payload = decode_token(token)
        redis_client.setex(
            f"blacklist:{payload['jti']}",
            timedelta(days=7),  # Keep blacklisted tokens for 7 days
            "1",
        )
    except Exception as e:
        logger.error("Token blacklisting failed", error=str(e))


class JWTBearer(HTTPBearer):
    """JWT bearer token authentication."""

    async def __call__(
        self, credentials: HTTPAuthorizationCredentials = Security(HTTPBearer())
    ) -> Dict[str, Any]:
        """Validate bearer token."""
        try:
            if not credentials:
                raise InvalidTokenError("Invalid authorization credentials")

            token = credentials.credentials
            payload = decode_token(token)

            return payload

        except InvalidTokenError as e:
            raise HTTPException(
                status_code=401, detail=str(e), headers={"WWW-Authenticate": "Bearer"}
            )


def rate_limit(requests: int, period: int):
    """
    Rate limiting decorator.

    Args:
        requests: Number of requests allowed
        period: Time period in seconds
    """

    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Get client IP
            request = kwargs.get("request")
            if not request:
                return await func(*args, **kwargs)

            client_ip = request.client.host

            # Check rate limit
            key = f"ratelimit:{client_ip}:{func.__name__}"
            current = redis_client.get(key)

            if current and int(current) >= requests:
                raise RateLimitExceededError(
                    f"Rate limit exceeded. Maximum {requests} requests per {period} seconds."
                )

            # Increment counter
            pipe = redis_client.pipeline()
            pipe.incr(key)
            pipe.expire(key, period)
            pipe.execute()

            return await func(*args, **kwargs)

        return wrapper

    return decorator


async def validate_image_file(file_bytes: bytes) -> None:
    """
    Validate an uploaded image file.

    Args:
        file_bytes: Raw file bytes

    Raises:
        FileValidationError: If validation fails
    """
    # Check file size
    if len(file_bytes) > MAX_UPLOAD_SIZE:
        raise FileValidationError(
            f"File size exceeds maximum allowed size of {MAX_UPLOAD_SIZE / 1024 / 1024:.1f}MB"
        )

    # Check file type
    file_type = magic.from_buffer(file_bytes, mime=True)
    if file_type not in ALLOWED_IMAGE_TYPES:
        raise FileValidationError(
            f"Invalid file type. Allowed types: {', '.join(ALLOWED_IMAGE_TYPES)}"
        )

    # Validate image dimensions
    try:
        image = Image.open(io.BytesIO(file_bytes))
        width, height = image.size
        if width < MIN_IMAGE_SIZE[0] or height < MIN_IMAGE_SIZE[1]:
            raise FileValidationError(
                f"Image dimensions must be at least {MIN_IMAGE_SIZE[0]}x{MIN_IMAGE_SIZE[1]} pixels"
            )
    except Exception as e:
        raise FileValidationError(f"Invalid image file: {str(e)}")

    # Calculate file hash
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    return file_hash


def get_current_user(token: Dict[str, Any] = Depends(JWTBearer())) -> Dict[str, Any]:
    """Get current authenticated user from token."""
    return token
