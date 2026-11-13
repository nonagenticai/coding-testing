# coding-testing

[![CI](https://github.com/nonagenticai/coding-testing/actions/workflows/ci.yml/badge.svg)](https://github.com/nonagenticai/coding-testing/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/nonagenticai/coding-testing)](https://github.com/nonagenticai/coding-testing/releases)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Multi-language test execution and AI-powered test analysis service for the code_graph_helper platform. Accepts source code and test code, runs tests in isolated Docker containers across 8+ languages, stores results, and provides AI-driven insights including pattern detection, failure prediction, and smart test selection.

## Architecture

```
coding-gateway:8687
  └─ /api/tests/* ──► coding-testing:8689
                           ├── Language-specific test executor services (Docker containers)
                           │     python-test-executor:8087
                           │     javascript-test-executor:8085
                           │     java-test-executor:8088
                           │     rust-test-executor:8089
                           │     csharp-test-executor:8090
                           │     go-test-executor:8091
                           │     cpp-test-executor:8094
                           ├── coding-containers:8691 (container validation)
                           ├── coding-ai:8690 (AI analysis)
                           ├── quality-analyzer:8086
                           ├── PostgreSQL (test results storage)
                           └── Redis (caching)
```

The service is the central orchestrator for all test-related operations. It does not run tests directly -- it delegates execution to language-specific executor services and manages the lifecycle, storage, and analysis of results.

**Key components:**

- **Routers** (`src/routers/`): Five router groups mounted under `/api/tests/*` -- testing (quality/discovery), test execution, test orchestration (suites/plans), intelligent testing (AI), and test results data.
- **Services** (`src/services/`): Business logic including code execution delegation, Docker management, AI failure analysis (PydanticAI agents), test orchestration, quality metrics, and test results storage.
- **Auth** (`src/auth/`): Multi-layer tenant isolation via X-Tenant-Id header, license introspection, and trusted-network bypass. All data operations are scoped to tenant.
- **Middleware** (`src/middleware/`): Auth (Istio header extraction + API key fallback), security, validation, CORS, and process-time metrics.
- **Processing** (`src/processing/`): Test execution pipeline -- result processing, hypothesis management, timeout management, HTTP test runner.
- **Clients** (`src/clients/`): HTTP clients for cross-service communication.
- **Shared** (`src/shared/`): Container validation, shared dependencies, response helpers, types.

## API Routes

All routes are prefixed by the gateway at `/api/tests`. Within this service, the prefix is `/api/tests` as well.

### Root / Health

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/` | No | Service info (name, version, docs link) |
| GET | `/health` | No | Root health check (Kubernetes probe target) |
| GET | `/api/health` | Tenant | Health check with uptime |
| GET | `/api/health/ready` | No | Kubernetes readiness probe |
| GET | `/api/health/live` | No | Kubernetes liveness probe |
| GET | `/api/health/detailed` | Tenant | Detailed health with system metrics (CPU, memory) |
| GET | `/api/health/system-metrics` | Tenant | System resource metrics (psutil) |

### Testing & Quality (`/api/tests`)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/tests/execute` | User+Tenant | Execute tests for code in any supported language |
| POST | `/api/tests/batch` | User+Tenant | Batch execute multiple tests in parallel |
| POST | `/api/tests/validate-security` | User+Tenant | Static security analysis of code before execution |
| POST | `/api/tests/quality-check` | User+Tenant | Run quality tools (ruff, ty, eslint, etc.) on code |
| POST | `/api/tests/quality-scan` | User+Tenant | Scan entire container codebase for quality issues |
| GET | `/api/tests/supported-tools` | User+Tenant | List available quality tools, optionally filtered by language |
| GET | `/api/tests/languages` | User+Tenant | List supported languages with test types and tools |
| GET | `/api/tests/languages/{language}/test-types` | User+Tenant | Get test types for a specific language |
| GET | `/api/tests/containers` | User+Tenant | List available test containers (all/test/provisioned) |
| GET | `/api/tests/health` | Tenant | Testing subsystem health (Docker, cache, quality tools) |
| GET | `/api/tests/metrics/summary` | User+Tenant | Aggregated test/quality metrics for a time period |

### Test Execution (`/api/tests/execute`)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/tests/execute/languages` | User+Tenant | Supported languages with container specs |
| GET | `/api/tests/execute/languages/{language}/test-types` | User+Tenant | Test types for a language |
| POST | `/api/tests/execute/execute` | User+Tenant | Execute tests in containerized environment |
| POST | `/api/tests/execute/batch` | User+Tenant | Batch execution with configurable parallelism |
| GET | `/api/tests/execute/active` | User+Tenant | List currently active test executions |
| DELETE | `/api/tests/execute/cleanup/{test_id}` | User+Tenant | Cleanup resources for a test execution |
| GET | `/api/tests/execute/container-specs` | User+Tenant | Container specifications per language |
| GET | `/api/tests/execute/health` | Tenant | Test execution subsystem health |

### Test Orchestration (`/api/tests/orchestration`)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/tests/orchestration/suites` | User+Tenant | Create a test suite (group of related tests for a container) |
| GET | `/api/tests/orchestration/suites` | User+Tenant | List test suites with optional filters |
| POST | `/api/tests/orchestration/suites/{suite_id}/execute` | User+Tenant | Execute all tests in a suite (async, parallel or serial) |
| GET | `/api/tests/orchestration/suites/{suite_id}/results` | User+Tenant | Get execution results for a suite |
| POST | `/api/tests/orchestration/plans` | User+Tenant | Create an orchestration plan (ordered suite execution with dependencies) |
| POST | `/api/tests/orchestration/plans/{plan_id}/execute` | User+Tenant | Execute an orchestration plan (async) |
| GET | `/api/tests/orchestration/execution-history` | User+Tenant | Get execution history with optional container filter |
| GET | `/api/tests/orchestration/metrics/summary` | User+Tenant | Orchestration metrics (success rates, language distribution) |

### Intelligent Testing - AI (`/api/tests/ai`)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/tests/ai/status` | Tenant | Module status for all intelligent testing components |
| GET | `/api/tests/ai/recommendations` | User+Tenant | AI-generated test optimization recommendations (container-scoped) |
| GET | `/api/tests/ai/patterns` | User+Tenant | Detected test execution patterns (container-scoped) |
| GET | `/api/tests/ai/supported-languages` | Tenant | Supported languages via intelligent testing engine |
| POST | `/api/tests/ai/execute` | User+Tenant | Execute test with AI optimization, pattern analysis, failure analysis |
| POST | `/api/tests/ai/execute/batch` | User+Tenant | Batch AI-optimized test execution |
| GET | `/api/tests/ai/insights` | User+Tenant | Comprehensive execution insights from AI analysis |
| GET | `/api/tests/ai/health` | User+Tenant | Intelligent testing service health |
| GET | `/api/tests/ai/settings` | Admin | Get service settings (admin only) |
| PUT | `/api/tests/ai/settings` | Admin | Update service settings (admin only) |
| POST | `/api/tests/ai/service/start` | Admin | Start intelligent test service |
| POST | `/api/tests/ai/service/stop` | Admin | Stop intelligent test service |
| POST | `/api/tests/ai/service/restart` | Admin | Restart intelligent test service |
| POST | `/api/tests/ai/pattern-analysis` | User+Tenant | AI pattern analysis on test results (container-scoped) |
| POST | `/api/tests/ai/smart-test-selection` | User+Tenant | AI-powered test selection based on code changes |
| POST | `/api/tests/ai/test-impact-analysis` | User+Tenant | Analyze which tests are impacted by code changes |
| POST | `/api/tests/ai/test-prioritization` | User+Tenant | Prioritize tests by impact, history, execution time |
| POST | `/api/tests/ai/advanced-recommendations` | User+Tenant | Advanced AI test recommendations (container-scoped) |
| POST | `/api/tests/ai/analyze` | Tenant | AI-powered test insight analysis |
| POST | `/api/tests/ai/comprehensive-analysis` | Tenant | Full AI analysis (patterns, performance, quality, predictions) |
| GET | `/api/tests/ai/metrics` | User+Tenant | AI test insights service metrics and capabilities |
| POST | `/api/tests/ai/recommendations` (insights) | User+Tenant | Generate recommendations from existing analysis |
| GET | `/api/tests/ai/analysis/{analysis_id}` | User+Tenant | Retrieve cached analysis results |
| GET | `/api/tests/ai/comprehensive-analysis/{analysis_id}` | User+Tenant | Retrieve cached comprehensive analysis |
| POST | `/api/tests/ai/predict-failures` | User+Tenant | Predict test failures from historical data and code changes |

### Test Results Data (`/api/tests`)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/tests/results/{container_name}/{test_id}` | User+Tenant | Get raw test execution data (for external AI analysis) |
| GET | `/api/tests/history/{container_name}` | User+Tenant | Historical test results with filters (name, language, success, pagination) |
| POST | `/api/tests/results/{container_name}/analyzed` | User+Tenant | Store externally-analyzed test results (from n8n, worker_management, etc.) |
| GET | `/api/tests/health` (results) | No | Test results data API health |

### Gateway Aliases

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/tests/status` | Tenant | Alias for `/api/tests/ai/status` |
| GET | `/api/tests/ready` | No | Alias for `/api/health/ready` |

### Behaviour parity (`/api/parity`)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/parity/{container_name}/record` | User+Tenant (write) | Run the cases against the SOURCE (returns a `recording`) |
| POST | `/api/parity/{container_name}/replay` | User+Tenant (write) | Run them against the TARGET, diff against `expected_*` or a recording |
| POST | `/api/parity/{container_name}/compare` | User+Tenant (write) | Source then target in one job, diff target vs source |
| GET | `/api/parity/{container_name}/runs/{run_id}` | User+Tenant | Poll an async run |

A side is an HTTP service by default (`kind: "service"`: fresh database, upstream stubs,
health wait). A **CLI or library** being ported (e.g. Python to Rust) uses `kind: "cli"`:
no database, stub or port; `start` is the argv PREFIX and each case's `args` are appended.
`migrate`, `upstreams` and `request_headers` are rejected on a cli side (422), and a corpus
never mixes cli cases with http/websocket/sse cases (422). Build the binary first (exec
`cargo build --release`); the parity run does not build.

A cli case (inline or one JSONL line):

```json
{"name": "add", "kind": "cli", "args": ["add", "2", "3"],
 "stdin": "optional text, or any JSON value (sent as JSON)",
 "env": {"OPTIONAL": "per-case env"}, "timeout": 30,
 "stdout_format": "text | json", "compare_stderr": false}
```

Per case the driver runs `start + args` with cwd = the side's `workdir` and records
`exit_code`, `stdout` (or `stdout_json` when `stdout_format: "json"`), `stderr`. Each stream
is read incrementally and capped at 1 MiB: past it the command's process group is killed and
the case is an error with `truncated: true` (+ `*_clipped`). A case `timeout` is at most 300 s
(larger is a 422); the default is the request's `request_timeout`, capped the same way. `{case_dir}` in `start`/`args` is a fresh empty directory
per case, the same path on every side. The diff compares `exit_code`, then stdout (as JSON
under `/stdout` honouring `ignore_paths`, else as text), and stderr only when
`compare_stderr` is true. A timeout, a command that cannot start, or a background process
that keeps a pipe open past the timeout is a case error (never passes). For `record`/`replay` against expectations: `expected_exit_code` (required),
`expected_stdout` or `expected_stdout_json`, `expected_stderr`.

```json
POST /api/parity/ws-1/compare
{"cases": [{"name": "add", "kind": "cli", "args": ["add", "2", "3"]}],
 "source": {"kind": "cli", "workdir": "py", "start": ["uv", "run", "mytool"]},
 "target": {"kind": "cli", "workdir": "rs", "start": ["target/release/mytool"]}}
```

**Library parity** is CLI parity through a call harness. coding-testing ships
`lib_call_harness.py` next to the driver; `{lib_harness}` in `start` is its path. Python
side: `start: ["uv", "run", "python", "{lib_harness}"]`, a case `args: ["pkg.mod:func"]`,
`stdin: {"args": [...], "kwargs": {...}}`, `stdout_format: "json"`. The Rust side is a
harness binary in the target repo (e.g. `src/bin/parity_harness.rs`, `start:
["target/release/parity_harness"]`) that implements the SAME contract, run from the same
corpus:

| | Contract |
|---|---|
| argv[1] | `"pkg.mod:func"` (the part after `:` is an attribute path, e.g. `Class.method`) |
| stdin | `{"args": [...], "kwargs": {...}}`, both optional; empty stdin = no arguments |
| stdout, exit 0 | exactly one JSON object: `{"return": <json>}` or `{"raises": "<ExcType>", "message": "<str>"}` |
| stdout, exit 2 | `{"error": "<str>"}`: the call could not be made, or the return value is not JSON |
| stderr | free (anything the function prints goes here); not compared unless `compare_stderr` |

`ExcType` is the bare class name (`ValueError`); `message` is `str(exc)`. A Rust harness maps
its error types onto the Python names the corpus expects. A `record` run's `recording`
(`coding-testing.behaviour-recording/1`) can be committed to the target as golden data.

## Supported Languages

Python, JavaScript, TypeScript, Go, Rust, Java, C#, C++

Each language has its own test executor service running in a dedicated Docker container. Test types vary by language (e.g., Python: unit, integration, coverage, security; Rust: unit, doc, benchmark, coverage, clippy, fmt).

## Environment Variables

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVICE_NAME` | `coding-testing` | Service name |
| `VERSION` | `1.0.0` | Service version |
| `ENVIRONMENT` | `production` | Environment (production/development/test) |
| `ENABLE_AUTH` | `true` | Enable authentication middleware |
| `DEFAULT_TENANT_ID` | `n8n-dev` | Fallback tenant for trusted-network requests |
| `API_KEY` | - | API key for dev/test auth fallback |
| `TRUSTED_CIDRS` | RFC1918 ranges | Trusted network CIDRs for auth bypass |

### Data Stores

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | - | PostgreSQL connection string |
| `REDIS_URL` | - | Redis connection string |

### Service URLs

| Variable | Default | Description |
|----------|---------|-------------|
| `PYTHON_TEST_SERVICE_URL` | `http://python-test-executor:8087` | Python executor |
| `JAVASCRIPT_TEST_SERVICE_URL` | `http://javascript-test-executor:8085` | JS executor |
| `JAVA_TEST_SERVICE_URL` | `http://java-test-executor:8088` | Java executor |
| `RUST_TEST_SERVICE_URL` | `http://rust-test-executor:8089` | Rust executor |
| `CSHARP_TEST_SERVICE_URL` | `http://csharp-test-executor:8090` | C# executor |
| `GO_TEST_SERVICE_URL` | `http://go-test-executor:8091` | Go executor |
| `CPP_TEST_SERVICE_URL` | `http://cpp-test-executor:8094` | C++ executor |
| `QUALITY_ANALYSIS_SERVICE_URL` | `http://quality-analyzer:8086` | Quality analyzer |
| `CONTAINERS_SERVICE_URL` | `http://coding-containers:8691` | Container management service |
| `AI_SERVICE_URL` | `http://coding-ai:8690` | AI analysis service |

### PydanticAI (AI-powered analysis)

| Variable | Default | Description |
|----------|---------|-------------|
| `PYDANTIC_AI_PROVIDER` | `openai` | LLM provider |
| `PYDANTIC_AI_MODEL` | `gpt-4` | Model name |
| `PYDANTIC_AI_API_KEY` | - | Provider API key |
| `PYDANTIC_AI_BASE_URL` | - | Custom provider base URL |
| `PYDANTIC_AI_MAX_RETRIES` | `3` | Max retries on failure |
| `PYDANTIC_AI_TIMEOUT` | `30` | Request timeout (seconds) |
| `PYDANTIC_AI_TEMPERATURE` | `0.1` | Model temperature |
| `PYDANTIC_AI_MAX_TOKENS` | `2000` | Max output tokens |
| `PYDANTIC_AI_RATE_LIMIT_RPM` | `100` | Rate limit (requests/min) |
| `PYDANTIC_AI_ENABLED` | `true` | Enable AI features |

### Docker Compose / OpenTelemetry

These are set in `docker-compose.yml` and consumed by the OpenTelemetry SDK, not directly by application code:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8689` | Listening port (Dockerfile CMD) |
| `LOG_LEVEL` | `INFO` | Logging level |
| `LOG_FORMAT` | `json` | Log output format |
| `OTLP_TRACES_ENDPOINT` | - | OpenTelemetry traces endpoint |
| `OTEL_SERVICE_NAME` | `coding-testing` | OTel service name |
| `OTEL_SDK_DISABLED` | `false` | Disable OTel SDK |

## Authentication

Four-layer authentication (evaluated in order, first match wins):

1. **Istio-forwarded headers** (`x-user-id`, `x-user-roles`, `x-tenant-id`): Set by the gateway/mesh after license-authz validation. If `x-user-id` is present, user identity is constructed from headers.
2. **Trusted network bypass**: RFC1918 clients (no `x-user-id` header) get an `internal` user with `DEFAULT_TENANT_ID` and `user` role.
3. **API key** (dev/test only): `Authorization: Bearer <API_KEY>` when `ENVIRONMENT` is `development` or `test`.
4. **Reject**: Returns 401 Unauthorized.

Public paths exempt from auth: `/health`, `/healthz`, `/ready`, `/readyz`, `/`, `/openapi.json`, `/docs`, `/redoc`, `/favicon.ico`.

Tenant isolation is enforced on all data-access endpoints via `X-Tenant-Id` header or license token introspection (`/license/introspect` on the licenses service).

## Directory Structure

```
src/
  main.py                    # Uvicorn entrypoint
  app_factory.py             # FastAPI app creation, middleware, router mounting
  auth/
    dependencies.py          # get_current_user, require_role, require_any_role
    models.py                # User model with tenant isolation
    tenant.py                # TenantMiddleware, require_tenant, license introspection
  clients/                   # HTTP clients for cross-service calls
  config/
    settings.py              # Settings class with env var defaults
    core_config_pydantic_ai.py  # PydanticAI LLM config from env
    core_config_settings.py  # Re-exports Settings
  exceptions/
    api_exceptions.py        # Custom API exception classes
  middleware/
    auth.py                  # AuthMiddleware (Istio headers, trusted network, API key)
    security.py              # SecurityMiddleware
    validation.py            # ValidationMiddleware
    metrics.py               # Process-time header
    exception_handlers.py    # Global exception handlers
  models/                    # Pydantic models (testing, AI analysis, errors, containers)
  processing/                # Test execution pipeline (result processor, timeout, HTTP runner)
  routers/
    health.py                # /api/health/* endpoints
    testing.py               # /api/tests/* (quality, discovery) - delegates to testing_modules/
    testing_modules/         # Modular sub-routers for testing
    test_execution.py        # /api/tests/execute/* endpoints
    test_orchestration.py    # /api/tests/orchestration/* endpoints
    test_results_data.py     # /api/tests/results/*, /api/tests/history/*
    intelligent_testing.py   # /api/tests/ai/* - delegates to intelligent_testing_modules/
    intelligent_testing_modules/  # 11 sub-routers for AI testing features
  services/                  # Business logic (execution, AI analysis, quality, storage)
  shared/                    # Container validation, response helpers, dependencies, types
```

## Dependencies

- **coding-containers** (8691): Container validation and listing. All container-scoped endpoints call this to verify ownership.
- **coding-ai** (8690): AI analysis capabilities.
- **Language test executors** (8085-8094): Actual test execution in isolated containers.
- **quality-analyzer** (8086): Quality analysis tools.
- **PostgreSQL**: Test results persistence.
- **Redis**: Result caching and session data.
- **licenses service** (8544): License token introspection for tenant resolution.

## Development

```bash
uv sync                          # install dependencies
uv run ruff check src/           # lint
uv run ruff format src/          # format
uv run ty check src/             # type check
uv run pytest                    # run tests (requires DATABASE_URL — see below)
uv run uvicorn main:app --port 8689 --reload  # run locally
```

### Local test database (required)

The test suite requires a real Postgres 18 and has **no SQLite fallback**. The
results/orchestration store speaks Postgres-only SQL (`jsonb_set`, `to_jsonb`,
`::jsonb`), so a SQLite stand-in runs different code and reports a different
result than CI. `pytest` aborts loudly if `DATABASE_URL` is unset or is not a
Postgres DSN. CI provisions a per-job ephemeral `postgres:18` and exports
`DATABASE_URL` (archetype `has_postgres: true` + `alembic: true`); locally:

```bash
docker run -d --name coding-testing-pg \
  -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=coding_testing_test \
  -p 127.0.0.1:5439:5432 postgres:18
export DATABASE_URL=postgresql+psycopg://postgres:postgres@127.0.0.1:5439/coding_testing_test  # pragma: allowlist secret
uv run alembic upgrade head      # create the tables
uv run pytest
```

Port 5439 matches the `sqlalchemy.url` default already in `alembic.ini`, so
`alembic` works against this container even with no env var exported.

Linting uses **ruff** (not black/isort). Type checking uses **ty** (not mypy). See `pyproject.toml` for config.

## CI/CD

GitHub Actions workflow: `ruff + ty → build amd64 → smoke test → push`

Image pushed to `192.168.1.114:5000/coding-testing:sha-<commit>`, promoted to `:latest` on success. Flux auto-deploys to K3s.

## Security

Report vulnerabilities privately through the repository's Security tab — see
[`SECURITY.md`](SECURITY.md). Do not open a public issue for a security problem.

## Contributing

Issues and pull requests are welcome. See
[`CONTRIBUTING.md`](https://github.com/nonagenticai/.github/blob/main/CONTRIBUTING.md) for the
organisation-wide rules; setup, test and lint commands are in this README. Commits carry DCO
sign-off.

## License

Apache-2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
