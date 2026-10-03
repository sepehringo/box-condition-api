# Hetzner deployment

Use an **x86 Linux VPS with at least 2 vCPUs and 4 GB RAM**, such as Ubuntu 24.04.
The image targets `linux/amd64`; this dependency set is not intended for an ARM VPS.
Allow extra disk space for image builds and temporary uploads. Two CPU inference
workers are configured; measurements must be repeated on the VPS.

## Server and DNS

Point an A record for your demo hostname to the VPS's public IPv4 address.
Only add an AAAA record if IPv6 is configured and reachable. In the Hetzner cloud
firewall, allow TCP 80 and 443 publicly and SSH from your own IP. Keep SSH available.
The Compose file publishes only Caddy's ports; API port 8000 stays internal.

Install Docker Engine and the Compose plugin using
[Docker's Ubuntu apt-repository instructions](https://docs.docker.com/engine/install/ubuntu/#install-using-the-apt-repository).
Enable Docker at boot with `sudo systemctl enable --now docker`.
The commands below assume permission to run Docker; prepend `sudo` if required.

## First launch

```sh
git clone https://github.com/sepehringo/box-condition-api.git
cd box-condition-api
cp .env.example .env
chmod 600 .env
openssl rand -hex 32
```

Edit `.env`: set `DEMO_DOMAIN` to your real hostname, without a scheme or path;
paste the generated random key into `BOX_API_KEY`, and set `BOX_IMAGE_TAG` to the
output of `git rev-parse --short HEAD`. The file is ignored by Git and Docker.
The placeholder key intentionally fails hosted startup. Share the real key
privately with reviewers; they enter it in Swagger's **Authorize** dialog.

```sh
docker compose build --pull
docker compose up -d --wait --wait-timeout 300
docker compose ps
curl --fail https://YOUR_HOSTNAME/ready
```

Caddy obtains and renews HTTPS certificates once DNS and ports are reachable.
Certificates persist in named volumes. Its proxy streams request bodies and
allows 240 seconds for response headers, covering the default upload, queue,
and processing budgets. See the [Caddy reverse proxy documentation](https://caddyserver.com/docs/caddyfile/directives/reverse_proxy).
Open `https://YOUR_HOSTNAME/docs` and submit the images in `samples/`.
Replace the demo placeholder in the main README with the real `/docs` URL after validation.

## Verify the deployment

From a machine with Python 3.11 and this repository, set `BOX_API_KEY` in the
shell environment and run:

```sh
python scripts/smoke_api.py --url https://YOUR_HOSTNAME
python scripts/benchmark_api.py --url https://YOUR_HOSTNAME \
  --image samples/box-1.jpg samples/box-2.jpg \
  --concurrency 1 5 10 25 40 --requests 40 --batch-size 2 --timeout 240
```

The smoke test checks public docs, health/readiness, missing/wrong key rejection,
ordered responses and four simultaneous batches against the actual model.
Benchmarking needs `requirements-dev.txt`. The benchmark reads the key from
`BOX_API_KEY` and never includes it in its report. Queue saturation waits;
the configured queue timeout can still return 504.

## Logs, restart and updates

```sh
docker compose logs --tail 100 api caddy
docker compose restart api
docker compose ps
```

Run the smoke test again after restarting. Containers restart after a process
failure or host reboot; a failing health check alone does not trigger a restart.
Temporary queues do not survive restarts. Uploads are ephemeral and cleaned when
work finishes; container recreation also discards its writable layer.

Before updating, record the current `BOX_IMAGE_TAG` from `.env`. Keep that image
locally. Pull the latest code, set a new commit-based tag in `.env`, then build:

```sh
git pull --ff-only
git rev-parse --short HEAD
# Set BOX_IMAGE_TAG in .env to the commit shown above.
docker compose build --pull api
docker compose up -d --wait --wait-timeout 300
```

Re-run the HTTPS smoke test. To roll back the API, set `BOX_IMAGE_TAG` to the
previous retained tag and run `docker compose up -d --no-build api`. If deployment
configuration also changed, restore its previous Git version before recreating
services. Keep `.env` and the Caddy certificate volumes. Do not prune retained
images until you no longer need them for rollback.

To rotate access, generate a new key, replace `BOX_API_KEY` in `.env`, and run
`docker compose up -d api`; distribute the replacement privately.
