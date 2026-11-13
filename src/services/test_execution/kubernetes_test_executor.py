"""
Kubernetes-based test executor.

Runs language test commands inside ephemeral Kubernetes Jobs instead of
Docker containers. Intended for in-cluster production deployments where
Docker is not available.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from typing import Any
from uuid import uuid7

from kubernetes import client, config
from kubernetes.client import ApiException

logger = logging.getLogger(__name__)


class KubernetesTestExecutor:
    """Execute tests using Kubernetes Jobs."""

    def __init__(self, namespace: str | None = None, in_cluster: bool | None = None) -> None:
        self.namespace = namespace or os.environ.get("PROVISIONER_NAMESPACE", "code-graph-helper")
        self.node_arch = os.environ.get("PROVISIONER_NODE_ARCH")
        self.pod_start_timeout = int(os.environ.get("K8S_TEST_POD_TIMEOUT_SECONDS", "120"))
        self.pod_completion_timeout = int(
            os.environ.get("K8S_TEST_COMPLETION_TIMEOUT_SECONDS", "900")
        )

        if in_cluster is None:
            in_cluster = os.environ.get("KUBERNETES_IN_CLUSTER", "false").lower() == "true"

        try:
            if in_cluster:
                config.load_incluster_config()
                logger.info(
                    "K8s test executor using in-cluster config (namespace=%s)",
                    self.namespace,
                )
            else:
                config.load_kube_config()
                logger.info(
                    "K8s test executor using local kubeconfig (namespace=%s)",
                    self.namespace,
                )
        except Exception as exc:
            logger.error("Failed to load Kubernetes configuration: %s", exc)
            raise

        self.batch = client.BatchV1Api()
        self.core = client.CoreV1Api()

    async def run(
        self, request: Any, container_spec: Any, test_command: list[str]
    ) -> dict[str, Any]:
        """Run the provided test command inside a Kubernetes Job."""

        job_name = f"test-exec-{request.language.value}-{uuid7().hex[:8]}"
        volume_name = "workspace"

        init_script = self._build_init_script(request)

        job = client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(
                name=job_name,
                namespace=self.namespace,
                labels={
                    "managed-by": "code-graph-helper",
                    "component": "test-executor",
                    "language": request.language.value,
                },
            ),
            spec=client.V1JobSpec(
                ttl_seconds_after_finished=600,
                template=client.V1PodTemplateSpec(
                    spec=client.V1PodSpec(
                        node_selector=(
                            {"kubernetes.io/arch": self.node_arch} if self.node_arch else None
                        ),
                        restart_policy="Never",
                        init_containers=[
                            client.V1Container(
                                name="init-workspace",
                                image="alpine:3.19",
                                command=["/bin/sh", "-c"],
                                args=[init_script],
                                volume_mounts=[
                                    client.V1VolumeMount(name=volume_name, mount_path="/workspace")
                                ],
                            )
                        ],
                        containers=[
                            client.V1Container(
                                name="runner",
                                image=container_spec.image,
                                command=test_command,
                                working_dir="/workspace",
                                env=[
                                    client.V1EnvVar(name=key, value=value)
                                    for key, value in container_spec.environment.items()
                                ],
                                volume_mounts=[
                                    client.V1VolumeMount(name=volume_name, mount_path="/workspace")
                                ],
                                resources=client.V1ResourceRequirements(
                                    limits={
                                        "cpu": container_spec.cpu_quota,
                                        "memory": container_spec.memory_limit,
                                    },
                                    requests={"cpu": "100m", "memory": "128Mi"},
                                ),
                            )
                        ],
                        volumes=[
                            client.V1Volume(
                                name=volume_name,
                                empty_dir=client.V1EmptyDirVolumeSource(),
                            )
                        ],
                    )
                ),
            ),
        )

        try:
            await asyncio.to_thread(self.batch.create_namespaced_job, self.namespace, job)
            logger.info("Created test executor Job: %s", job_name)

            pod_name = await self._wait_for_pod(job_name)
            exit_code = await self._wait_for_completion(pod_name)
            logs = await asyncio.to_thread(
                self.core.read_namespaced_pod_log, pod_name, self.namespace
            )

            return {
                "job_name": job_name,
                "pod_name": pod_name,
                "exit_code": exit_code,
                "stdout": logs,
                "stderr": "" if exit_code == 0 else logs,
            }
        finally:
            # Best-effort cleanup; ignore failures so tests still return
            try:
                await asyncio.to_thread(
                    self.batch.delete_namespaced_job,
                    job_name,
                    self.namespace,
                    client.V1DeleteOptions(propagation_policy="Background"),
                )
                logger.debug("Deleted Job %s", job_name)
            except ApiException as exc:
                if exc.status != 404:
                    logger.warning("Failed to delete Job %s: %s", job_name, exc)

    def _build_init_script(self, request: Any) -> str:
        """Create shell script that writes source and test files into /workspace."""

        commands: list[str] = ["set -e"]

        if request.source_code:
            filename = f"main{self._extension(request.language.value)}"
            commands.append(self._write_file_cmd(filename, request.source_code))

        for name, content in request.test_files.items():
            commands.append(self._write_file_cmd(name, content))

        for filename, content in self._language_config_files(request):
            commands.append(self._write_file_cmd(filename, content))

        return "\n".join(commands)

    def _write_file_cmd(self, filename: str, content: str) -> str:
        encoded = base64.b64encode(content.encode()).decode()
        # Use base64 to preserve newlines and quotes safely
        return f"echo {encoded} | base64 -d > /workspace/{filename}"

    async def _wait_for_pod(self, job_name: str) -> str:
        """Wait until the Job spawns a pod and return its name, with timeout."""

        async def _poll() -> str:
            while True:
                pods = await asyncio.to_thread(
                    self.core.list_namespaced_pod,
                    self.namespace,
                    label_selector=f"job-name={job_name}",
                )
                items = pods.items or []
                if items:
                    return items[0].metadata.name
                await asyncio.sleep(1)

        try:
            return await asyncio.wait_for(_poll(), timeout=self.pod_start_timeout)
        except TimeoutError as exc:
            raise TimeoutError(
                f"Pod for job {job_name} did not start within {self.pod_start_timeout}s"
            ) from exc

    async def _wait_for_completion(self, pod_name: str) -> int:
        """Wait for the pod to terminate and return its exit code, with timeout."""

        async def _poll() -> int:
            while True:
                pod = await asyncio.to_thread(
                    self.core.read_namespaced_pod_status,
                    pod_name,
                    self.namespace,
                )
                statuses = pod.status.container_statuses or []
                if statuses and statuses[0].state and statuses[0].state.terminated:
                    exit_code = statuses[0].state.terminated.exit_code
                    return exit_code if exit_code is not None else 1
                await asyncio.sleep(1)

        try:
            return await asyncio.wait_for(_poll(), timeout=self.pod_completion_timeout)
        except TimeoutError as exc:
            raise TimeoutError(
                f"Test pod {pod_name} did not complete within {self.pod_completion_timeout}s"
            ) from exc

    def _extension(self, language: str) -> str:
        return {
            "go": ".go",
            "rust": ".rs",
            "java": ".java",
            "csharp": ".cs",
            "cpp": ".cpp",
            "python": ".py",
            "javascript": ".js",
            "typescript": ".ts",
        }.get(language, ".txt")

    def _language_config_files(self, request: Any) -> list[tuple[str, str]]:
        """Return language-specific config files to write into the workspace."""
        files: list[tuple[str, str]] = []

        if request.language.value == "go":
            files.append(("go.mod", self._go_mod_content(request.dependencies)))

        if request.language.value == "rust":
            files.append(("Cargo.toml", self._cargo_toml_content(request.dependencies)))

        if request.language.value in {"javascript", "typescript"}:
            files.append(("package.json", self._package_json_content(request)))

        return files

    def _go_mod_content(self, dependencies: dict[str, str]) -> str:
        return f"""module test-module

go 1.21

{self._format_go_dependencies(dependencies)}
"""

    def _cargo_toml_content(self, dependencies: dict[str, str]) -> str:
        return f"""[package]
name = "test-crate"
version = "0.1.0"
edition = "2021"

[dependencies]
{self._format_rust_dependencies(dependencies)}
"""

    def _package_json_content(self, request: Any) -> str:
        package_json: dict[str, Any] = {
            "name": "test-project",
            "version": "1.0.0",
            "scripts": {
                "test": (
                    "jest" if request.language.value == "javascript" else "jest --preset ts-jest"
                )
            },
            "devDependencies": {"jest": "^29.0.0", **(request.dependencies or {})},
        }

        if request.language.value == "typescript":
            ts_deps: dict[str, str] = {
                "ts-jest": "^29.0.0",
                "@types/jest": "^29.0.0",
                "typescript": "^5.0.0",
            }
            package_json["devDependencies"].update(ts_deps)

        return json.dumps(package_json, indent=2)

    def _format_go_dependencies(self, dependencies: dict[str, str]) -> str:
        if not dependencies:
            return ""
        return "\n".join([f"require {dep} {version}" for dep, version in dependencies.items()])

    def _format_rust_dependencies(self, dependencies: dict[str, str]) -> str:
        if not dependencies:
            return ""
        return "\n".join([f'{dep} = "{version}"' for dep, version in dependencies.items()])
