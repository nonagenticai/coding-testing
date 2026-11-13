# Operation ID Plan: coding-testing

## Files to Edit

- `src/app_factory.py` (app-level routes: `/health`, `/`, `/api/tests/status`, `/api/tests/ready`)
- `src/routers/health.py` (prefix `/api/health`)
- `src/routers/testing_modules/test_execution.py` (prefix `/api/tests` via testing.py)
- `src/routers/testing_modules/test_discovery.py` (prefix `/api/tests` via testing.py)
- `src/routers/testing_modules/quality_analysis.py` (prefix `/api/tests` via testing.py)
- `src/routers/testing_modules/health_and_metrics.py` (prefix `/api/tests` via testing.py)
- `src/routers/test_execution.py` (prefix `/api/tests/execute`)
- `src/routers/test_orchestration.py` (prefix `/api/tests/orchestration`)
- `src/routers/test_results_data.py` (prefix `/api/tests`)
- `src/routers/intelligent_testing.py` (prefix `/api/tests/ai` -- status endpoint only)
- `src/routers/intelligent_testing_modules/discovery_and_recommendations.py` (prefix `/api/tests/ai`)
- `src/routers/intelligent_testing_modules/execution_coordination.py` (prefix `/api/tests/ai`)
- `src/routers/intelligent_testing_modules/analysis_and_insights.py` (prefix `/api/tests/ai`)
- `src/routers/intelligent_testing_modules/configuration_management.py` (prefix `/api/tests/ai`)
- `src/routers/intelligent_testing_modules/service_lifecycle.py` (prefix `/api/tests/ai`)
- `src/routers/intelligent_testing_modules/container_scoped_testing.py` (prefix `/api/tests/ai`)
- `src/routers/intelligent_testing_modules/ai_analysis_endpoints.py` (prefix `/api/tests/ai`)
- `src/routers/intelligent_testing_modules/ai_test_insights_config.py` (prefix `/api/tests/ai`)
- `src/routers/intelligent_testing_modules/insight_generation_services.py` (prefix `/api/tests/ai`)
- `src/routers/intelligent_testing_modules/report_formatting_endpoints.py` (prefix `/api/tests/ai`)
- `src/routers/intelligent_testing_modules/test_pattern_recognition.py` (prefix `/api/tests/ai`)

## Mapping Table

### app_factory.py (app-level routes, no prefix)

| Method | Effective Path | Function | operation_id |
|--------|---------------|----------|--------------|
| GET | `/health` | `root_health` | `health_check` |
| GET | `/` | `root` | `root` |
| GET | `/api/tests/status` | `gateway_alias_status` | `gateway_alias_status` |
| GET | `/api/tests/ready` | `gateway_alias_ready` | `gateway_alias_ready` |

### health.py (prefix `/api/health`)

| Method | Effective Path | Function | operation_id |
|--------|---------------|----------|--------------|
| GET | `/api/health` | `health_check` | `health_api_check` |
| GET | `/api/health/ready` | `service_readiness_check` | `health_ready` |
| GET | `/api/health/live` | `service_liveness_check` | `health_live` |
| GET | `/api/health/detailed` | `detailed_health_check` | `health_detailed` |
| GET | `/api/health/system-metrics` | `get_system_metrics` | `get_health_system_metrics` |

### testing_modules/test_execution.py (prefix `/api/tests`)

| Method | Effective Path | Function | operation_id |
|--------|---------------|----------|--------------|
| POST | `/api/tests/execute` | `execute_test` | `execute_test` |
| POST | `/api/tests/batch` | `execute_batch` | `execute_batch_tests` |
| POST | `/api/tests/validate-security` | `validate_code_security` | `validate_code_security` |

### testing_modules/test_discovery.py (prefix `/api/tests`)

| Method | Effective Path | Function | operation_id |
|--------|---------------|----------|--------------|
| GET | `/api/tests/supported-tools` | `get_supported_tools` | `list_supported_tools` |
| GET | `/api/tests/languages` | `get_supported_languages` | `list_supported_languages` |
| GET | `/api/tests/languages/{language}/test-types` | `get_language_test_types` | `get_language_test_types` |
| GET | `/api/tests/containers` | `list_test_containers` | `list_test_containers` |

### testing_modules/quality_analysis.py (prefix `/api/tests`)

| Method | Effective Path | Function | operation_id |
|--------|---------------|----------|--------------|
| POST | `/api/tests/quality-check` | `run_quality_check` | `run_quality_check` |
| POST | `/api/tests/quality-scan` | `run_quality_scan` | `run_quality_scan` |

### testing_modules/health_and_metrics.py (prefix `/api/tests`)

| Method | Effective Path | Function | operation_id |
|--------|---------------|----------|--------------|
| GET | `/api/tests/health` | `testing_health_check` | `testing_health_check` |
| GET | `/api/tests/metrics/summary` | `get_testing_metrics_summary` | `get_testing_metrics_summary` |

### test_execution.py (prefix `/api/tests/execute`)

| Method | Effective Path | Function | operation_id |
|--------|---------------|----------|--------------|
| GET | `/api/tests/execute/languages` | `get_supported_languages` | `list_execution_languages` |
| GET | `/api/tests/execute/languages/{language}/test-types` | `get_test_types_for_language` | `get_execution_language_test_types` |
| POST | `/api/tests/execute/execute` | `execute_tests` | `execute_containerized_tests` |
| POST | `/api/tests/execute/batch` | `execute_batch_tests` | `execute_containerized_batch_tests` |
| GET | `/api/tests/execute/active` | `get_active_executions` | `list_active_executions` |
| DELETE | `/api/tests/execute/cleanup/{test_id}` | `cleanup_test_execution` | `cleanup_test_execution` |
| GET | `/api/tests/execute/container-specs` | `get_container_specifications` | `list_container_specs` |
| GET | `/api/tests/execute/health` | `test_execution_health` | `test_execution_health_check` |

### test_orchestration.py (prefix `/api/tests/orchestration`)

| Method | Effective Path | Function | operation_id |
|--------|---------------|----------|--------------|
| POST | `/api/tests/orchestration/suites` | `create_test_suite` | `create_test_suite` |
| GET | `/api/tests/orchestration/suites` | `list_test_suites` | `list_test_suites` |
| POST | `/api/tests/orchestration/suites/{suite_id}/execute` | `execute_test_suite` | `execute_test_suite` |
| GET | `/api/tests/orchestration/suites/{suite_id}/results` | `get_suite_results` | `get_test_suite_results` |
| POST | `/api/tests/orchestration/plans` | `create_orchestration_plan` | `create_orchestration_plan` |
| POST | `/api/tests/orchestration/plans/{plan_id}/execute` | `execute_orchestration_plan` | `execute_orchestration_plan` |
| GET | `/api/tests/orchestration/execution-history` | `get_execution_history` | `list_orchestration_history` |
| GET | `/api/tests/orchestration/metrics/summary` | `get_orchestration_metrics` | `get_orchestration_metrics_summary` |

### test_results_data.py (prefix `/api/tests`)

| Method | Effective Path | Function | operation_id |
|--------|---------------|----------|--------------|
| GET | `/api/tests/results/{container_name}/{test_id}` | `get_raw_test_result` | `get_raw_test_result` |
| GET | `/api/tests/history/{container_name}` | `get_test_history` | `get_test_history` |
| POST | `/api/tests/results/{container_name}/analyzed` | `store_analyzed_result` | `store_analyzed_test_result` |
| GET | `/api/tests/health` | `health_check` | `test_results_data_health_check` |

### intelligent_testing.py (prefix `/api/tests/ai`)

| Method | Effective Path | Function | operation_id |
|--------|---------------|----------|--------------|
| GET | `/api/tests/ai/status` | `get_module_status` | `get_ai_module_status` |

### intelligent_testing_modules/discovery_and_recommendations.py (prefix `/api/tests/ai`)

| Method | Effective Path | Function | operation_id |
|--------|---------------|----------|--------------|
| GET | `/api/tests/ai/recommendations` | `get_recommendations` | `get_ai_recommendations` |
| GET | `/api/tests/ai/patterns` | `get_execution_patterns` | `get_execution_patterns` |
| GET | `/api/tests/ai/supported-languages` | `get_supported_languages` | `list_ai_supported_languages` |

### intelligent_testing_modules/execution_coordination.py (prefix `/api/tests/ai`)

| Method | Effective Path | Function | operation_id |
|--------|---------------|----------|--------------|
| POST | `/api/tests/ai/execute` | `execute_test` | `execute_intelligent_test` |
| POST | `/api/tests/ai/execute/batch` | `execute_batch_tests` | `execute_intelligent_batch_tests` |

### intelligent_testing_modules/analysis_and_insights.py (prefix `/api/tests/ai`)

| Method | Effective Path | Function | operation_id |
|--------|---------------|----------|--------------|
| GET | `/api/tests/ai/insights` | `get_execution_insights` | `get_execution_insights` |
| GET | `/api/tests/ai/health` | `get_health_status` | `get_ai_health_status` |

### intelligent_testing_modules/configuration_management.py (prefix `/api/tests/ai`)

| Method | Effective Path | Function | operation_id |
|--------|---------------|----------|--------------|
| GET | `/api/tests/ai/settings` | `get_service_settings` | `get_ai_service_settings` |
| PUT | `/api/tests/ai/settings` | `update_service_settings` | `update_ai_service_settings` |

### intelligent_testing_modules/service_lifecycle.py (prefix `/api/tests/ai`)

| Method | Effective Path | Function | operation_id |
|--------|---------------|----------|--------------|
| POST | `/api/tests/ai/service/start` | `start_service` | `start_ai_service` |
| POST | `/api/tests/ai/service/stop` | `stop_service` | `stop_ai_service` |
| POST | `/api/tests/ai/service/restart` | `restart_service` | `restart_ai_service` |

### intelligent_testing_modules/container_scoped_testing.py (prefix `/api/tests/ai`)

| Method | Effective Path | Function | operation_id |
|--------|---------------|----------|--------------|
| POST | `/api/tests/ai/pattern-analysis` | `analyze_test_patterns` | `analyze_test_patterns` |
| POST | `/api/tests/ai/smart-test-selection` | `smart_test_selection` | `smart_test_selection` |
| POST | `/api/tests/ai/test-impact-analysis` | `test_impact_analysis` | `analyze_test_impact` |
| POST | `/api/tests/ai/test-prioritization` | `test_prioritization` | `prioritize_tests` |
| POST | `/api/tests/ai/advanced-recommendations` | `advanced_test_recommendations` | `get_advanced_recommendations` |

### intelligent_testing_modules/ai_analysis_endpoints.py (prefix `/api/tests/ai`)

| Method | Effective Path | Function | operation_id |
|--------|---------------|----------|--------------|
| POST | `/api/tests/ai/analyze` | `analyze_test_insights` | `analyze_test_insights` |
| POST | `/api/tests/ai/comprehensive-analysis` | `comprehensive_test_analysis` | `run_comprehensive_analysis` |

### intelligent_testing_modules/ai_test_insights_config.py (prefix `/api/tests/ai`)

| Method | Effective Path | Function | operation_id |
|--------|---------------|----------|--------------|
| GET | `/api/tests/ai/health` | `health_check` | `ai_test_insights_health_check` |
| GET | `/api/tests/ai/metrics` | `get_service_metrics` | `get_ai_service_metrics` |

### intelligent_testing_modules/insight_generation_services.py (prefix `/api/tests/ai`)

| Method | Effective Path | Function | operation_id |
|--------|---------------|----------|--------------|
| POST | `/api/tests/ai/recommendations` | `generate_test_recommendations` | `generate_test_recommendations` |

### intelligent_testing_modules/report_formatting_endpoints.py (prefix `/api/tests/ai`)

| Method | Effective Path | Function | operation_id |
|--------|---------------|----------|--------------|
| GET | `/api/tests/ai/analysis/{analysis_id}` | `get_cached_analysis` | `get_cached_analysis` |
| GET | `/api/tests/ai/comprehensive-analysis/{analysis_id}` | `get_cached_comprehensive_analysis` | `get_cached_comprehensive_analysis` |

### intelligent_testing_modules/test_pattern_recognition.py (prefix `/api/tests/ai`)

| Method | Effective Path | Function | operation_id |
|--------|---------------|----------|--------------|
| POST | `/api/tests/ai/predict-failures` | `predict_test_failures` | `predict_test_failures` |

## Total: 69 routes

## Execution Steps

```bash
cd /Users/mini1/Movies/code_graph_helper/coding-testing && git pull --rebase
```

1. Add `operation_id=` to every `@router.get/post/put/delete/patch(` decorator and `@app.get/post(` decorator in all files listed above
2. Validate uniqueness -- ensure no duplicate `operation_id` values across the entire app
3. Run pre-commit checks:
   ```bash
   pre-commit run --all-files
   ```
4. Run tests:
   ```bash
   pytest
   ```
5. Commit and push:
   ```bash
   git add -A && git commit -m "Add explicit operation_id to all route decorators" && git push
   ```
