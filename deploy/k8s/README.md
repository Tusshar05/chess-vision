# Kubernetes deployment

Manifests: namespace, ConfigMap, two PVCs (models, history), API Deployment + Service + HPA,
dashboard Deployment + Service, and an optional Job that loads model files into the models PVC.

Images come from GHCR (pushed by CI on `main`): `ghcr.io/tusshar05/chess-vision-api` and
`...-dashboard`. For a local cluster build and load them instead:

```bash
docker build -f services/api/Dockerfile -t ghcr.io/tusshar05/chess-vision-api:latest .
docker build -f services/dashboard/Dockerfile -t ghcr.io/tusshar05/chess-vision-dashboard:latest .
```

## kind

```bash
kind create cluster --name chessvision
kind load docker-image ghcr.io/tusshar05/chess-vision-api:latest --name chessvision
kind load docker-image ghcr.io/tusshar05/chess-vision-dashboard:latest --name chessvision
kubectl apply -k deploy/k8s
# models: edit MODELS_URL_BASE in load-models-job.yaml (or kubectl cp), then
kubectl apply -f deploy/k8s/load-models-job.yaml
kubectl -n chessvision rollout restart deploy/chessvision-api
kubectl -n chessvision port-forward svc/chessvision-api 8000:8000 &
kubectl -n chessvision port-forward svc/chessvision-dashboard 8501:8501
```

## minikube

```bash
minikube start
minikube addons enable metrics-server        # required by the HPA
minikube image load ghcr.io/tusshar05/chess-vision-api:latest
minikube image load ghcr.io/tusshar05/chess-vision-dashboard:latest
kubectl apply -k deploy/k8s
minikube service -n chessvision chessvision-dashboard
```

## Validate without a cluster

```bash
kubeconform -strict -summary -ignore-missing-schemas deploy/k8s/*.yaml
```

(`make k8s-validate` does the same.)

## Notes / honest limitations

- **History + HPA:** SQLite on a ReadWriteOnce PVC only works for one replica. The HPA can scale the
  API to 4 pods, but each would need its own history; for real multi-replica use set `DB_URL` to
  Postgres. Recognition itself is stateless and scales fine.
- Readiness waits for `/health`; with no model files the API reports `degraded` and `/v1/recognize`
  returns 503 until the models PVC is populated.
- Not tested on a live cluster here - manifests are schema-validated only (kubeconform).
