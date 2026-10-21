PROJECT_CONTEXT = (
    "Project: E-commerce platform feature — 'User Wishlist'. "
    "Tech stack: Python 3.11, FastAPI, PostgreSQL, SQLAlchemy, pytest. "
    "Requirements: Users can add/remove products to a wishlist, view their wishlist, "
    "and receive notifications when a wishlisted item goes on sale. "
    "The feature must include API endpoints, database schema, unit tests, "
    "security review, and documentation."
)

REPO_SUMMARY = (
    "Repository structure: src/api/ (FastAPI routers), src/models/ (SQLAlchemy models), "
    "src/services/ (business logic), src/schemas/ (Pydantic schemas), "
    "tests/ (pytest), docs/ (API documentation). "
    "Existing models: User, Product, Order. Auth: JWT tokens."
)

ARCHITECT_PROMPT = (
    "You are the system architect. Design the Wishlist feature: "
    "define the data model, API endpoints, service layer interfaces, "
    "and integration points with existing User and Product models. "
    "Output a structured technical design document."
)

ARCHITECT_INTERFACES_PROMPT = (
    "Based on the high-level design, define detailed component interfaces: "
    "method signatures for WishlistService (add, remove, get, check, notify), "
    "expected input/output types, dependency injection boundaries, "
    "and the event-driven notification contract for sale alerts."
)

ARCHITECT_INTEGRATION_PROMPT = (
    "Based on the design and interfaces, create the integration plan: "
    "component initialization order, configuration requirements, "
    "database migration sequencing, feature flag strategy, "
    "and rollback procedure if deployment fails."
)

DB_DESIGNER_PROMPT = (
    "You are the database designer. Based on the architect's design, "
    "create the SQLAlchemy model for the Wishlist table, "
    "including foreign keys to User and Product, timestamps, "
    "and the Alembic migration script."
)

DB_MIGRATION_PROMPT = (
    "Based on the schema you designed, write the Alembic migration: "
    "upgrade and downgrade functions, composite indexes on (user_id, product_id), "
    "and any data backfill steps for existing users."
)

API_DESIGNER_PROMPT = (
    "You are the API designer. Based on the architect's design, "
    "define the FastAPI router with endpoints: POST /wishlist/add, "
    "DELETE /wishlist/remove, GET /wishlist, GET /wishlist/check/{product_id}. "
    "Include Pydantic request/response schemas."
)

API_VALIDATION_PROMPT = (
    "Based on the API endpoints, define error handling: "
    "HTTP status codes for each failure mode (404 not found, 409 duplicate, "
    "403 forbidden), validation error response format, "
    "and rate limiting headers (X-RateLimit-Remaining)."
)

BACKEND_CODER_PROMPT = (
    "You are the backend developer. Implement the Wishlist service layer: "
    "add_to_wishlist(), remove_from_wishlist(), get_wishlist(), "
    "check_in_wishlist(). Use the DB model and wire up the API router."
)

BACKEND_ROUTING_PROMPT = (
    "Wire up the FastAPI router: connect each endpoint to the WishlistService, "
    "add Depends() for DB session and current_user injection, "
    "configure authentication middleware, and add response_model annotations."
)

BACKEND_ERROR_HANDLING_PROMPT = (
    "Implement error handling for the Wishlist service: "
    "WishlistNotFoundError, DuplicateWishlistError exception classes, "
    "FastAPI exception handlers, structured logging for each error path, "
    "and graceful degradation when the notification service is unavailable."
)

TEST_WRITER_PROMPT = (
    "You are the test engineer. Write pytest tests for the Wishlist feature: "
    "test each API endpoint, test service layer functions, "
    "test edge cases (duplicate add, remove nonexistent, empty wishlist). "
    "Use fixtures for database session and test user."
)

TEST_EDGE_CASES_PROMPT = (
    "Write additional edge-case tests: concurrent wishlist modifications, "
    "wishlist with maximum items (1000), adding an out-of-stock product, "
    "removing the last item, and Unicode product names in wishlist entries."
)

TEST_INTEGRATION_PROMPT = (
    "Write integration tests: full API flow from request to response, "
    "test JWT authentication middleware rejects unauthenticated requests, "
    "test database transaction rollback on service failure, "
    "and test the sale notification trigger end-to-end."
)

SECURITY_REVIEWER_PROMPT = (
    "You are the security reviewer. Audit the Wishlist feature for: "
    "authentication/authorization (users can only access own wishlist), "
    "input validation, SQL injection prevention, rate limiting considerations, "
    "and IDOR vulnerabilities. Report findings and fixes."
)

SECURITY_INPUT_VALIDATION_PROMPT = (
    "Deep-dive into input validation: review all Pydantic schemas for "
    "injection vulnerabilities, verify query parameter sanitization, "
    "test for path traversal in product_id lookups, "
    "and check integer overflow on wishlist item counts."
)

SECURITY_RATE_LIMIT_PROMPT = (
    "Assess rate limiting and abuse prevention: evaluate per-user "
    "rate limits for wishlist operations, check for enumeration attacks "
    "on the check endpoint, verify CORS configuration, "
    "and review the notification endpoint for spam potential."
)

DOC_WRITER_PROMPT = (
    "You are the documentation writer. Write API documentation for the Wishlist "
    "endpoints: request/response examples, authentication requirements, "
    "error codes, and a quick-start guide for frontend integration."
)

DOC_QUICKSTART_PROMPT = (
    "Write a quick-start integration guide: step-by-step frontend "
    "instructions with JavaScript fetch() examples for each endpoint, "
    "WebSocket setup for real-time sale notifications, "
    "and common troubleshooting tips (401, 409, 429 errors)."
)

INTEGRATOR_PROMPT = (
    "You are the integration lead. Review all outputs from the team: "
    "verify the DB model matches the API schemas, tests cover the service layer, "
    "security findings are addressed, and documentation is accurate. "
    "Produce a final integration report with any remaining issues."
)

INTEGRATOR_REPORT_PROMPT = (
    "Produce the final deployment checklist: verify each interface contract "
    "between DB schema and API schemas, confirm all security findings "
    "have corresponding test coverage, list environment variables needed, "
    "and specify the migration execution order for production rollout."
)
