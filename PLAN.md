# coding-testing — Extraction Plan

## Purpose
Owns test execution, orchestration, intelligent testing, and test result data. Communicates with test executors (8085-8094) and coding-ai for test analysis.

## Port: 8689

---

## Endpoints

| Route Prefix | Source Router | Source Endpoint File |
|---|---|---|
| `/api/tests` | `testing` | `src/app/api/endpoints/testing.py` |
| `/api/tests/execute` | `test_execution` | `src/app/api/endpoints/test_execution.py` |
| `/api/tests/orchestration` | `test_orchestration` | `src/app/api/endpoints/test_orchestration.py` |
| `/api/tests/ai` | `intelligent_testing` | `src/app/api/endpoints/intelligent_testing.py` |
| `/api/tests` (data) | `test_results_data` | `src/app/api/endpoints/test_results_data.py` |

## Endpoint Sub-modules

| Sub-package | Files | Notes |
|---|---|---|
| `testing_modules/` | ~8 files | Testing endpoint helpers |
| `intelligent_testing_modules/` | ~6 files | AI-powered testing helpers |

## Services to Extract

| Monolith Service | Target Location |
|---|---|
| `test_execution/*` | `src/services/test_execution/` |
| `testing/*` | `src/services/testing/` |
| `services/test_failure_prediction_service.py` | `src/services/test_failure_prediction.py` |
| `processing/test_execution/*` | `src/processing/test_execution/` |

## Communication with Test Executors

| Executor | Port | URL Env Var |
|---|---|---|
| python-test-executor | 8087 | `PYTHON_TEST_SERVICE_URL` |
| javascript-test-executor | 8085 | `JAVASCRIPT_TEST_SERVICE_URL` |
| quality-analyzer | 8086 | `QUALITY_ANALYSIS_SERVICE_URL` |
| java-test-executor | 8088 | `JAVA_TEST_SERVICE_URL` |
| rust-test-executor | 8089 | `RUST_TEST_SERVICE_URL` |
| csharp-test-executor | 8090 | `CSHARP_TEST_SERVICE_URL` |
| go-test-executor | 8091 | `GO_TEST_SERVICE_URL` |
| cpp-test-executor | 8094 | `CPP_TEST_SERVICE_URL` |

## Dependencies
- coding-client (shared models, auth, DB)
- coding-ai client (for AI-powered test analysis)
- httpx (calls to test executors)
