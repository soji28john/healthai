from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
import re, json, logging

BLOCK_PATTERNS = [
    r"\b(kill myself|suicide|self.harm|overdose)\b",
    r"\b(bomb|weapon|explosive)\b",
]
ESCALATE_PATTERNS = [
    r"\b(chest pain|can't breathe|stroke|unconscious|emergency)\b",
]

class PolicyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method == "POST" and "/chat" in request.url.path:
            body = await request.body()
            try:
                text = json.loads(body).get("message", "").lower()
            except Exception:
                text = ""

            for pattern in BLOCK_PATTERNS:
                if re.search(pattern, text, re.IGNORECASE):
                    return self._crisis_response()

            request.state.escalate = any(
                re.search(p, text, re.IGNORECASE) for p in ESCALATE_PATTERNS
            )
            logging.info(f"policy: allow | escalate={request.state.escalate}")

        return await call_next(request)

    def _crisis_response(self):
        from starlette.responses import JSONResponse
        return JSONResponse(
            status_code=200,
            content={
                "response": "I'm concerned about what you've shared. Please contact a crisis helpline immediately.",
                "resources": ["International: befrienders.org", "US: 988 Suicide & Crisis Lifeline"],
                "policy_action": "BLOCKED"
            }
        )