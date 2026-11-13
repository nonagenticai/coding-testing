Real Kubernetes objects, read (get-only) from the strato cluster on 2026-10-29 and
stripped to what k8s_jobs reads (no env, args, command, volumes or IPs):

- `job-deadline-exceeded.json`: a Job that failed with DeadlineExceeded (its pod is
  already gone), from
  `kubectl get jobs -A -o json` (actions-runners/ci-self-heal-codeload-29885830).
- `pod-container-error.json`: a pod whose container last terminated with reason
  Error, exit 1, from `kubectl get pods -A -o json` (flux-system/source-controller-*).
  `kubectl get pods -A --field-selector=status.phase=Failed -o json` returned no pods
  at capture time, and no pod was OOMKilled or waiting on an image pull.
