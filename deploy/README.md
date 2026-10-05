# Hetzner deployment

Use an **x86 Linux VPS with at least 2 vCPUs and 4 GB RAM**, such as Ubuntu 24.04.
The image targets `linux/amd64`; this dependency set is not intended for an ARM VPS.
For automatic releases, use [the CI/CD setup below](#automatic-cicd-deployment).
The initial sections describe the alternative manual build workflow.
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

## Automatic CI/CD deployment

The workflow runs checks for every push and pull request. Successful `main` runs
publish the exact smoke-tested image to GHCR using a commit tag, then deploy by
digest when `DEPLOY_ENABLED=true`. Publication uses the workflow's `GITHUB_TOKEN`;
see [GitHub's registry documentation](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry).
The server pulls images instead of rebuilding them. Deployments are serialized
in GitHub and locked on the server. Expect brief downtime during container replacement.

### One-time server setup

Use the DNS/firewall setup above, Docker Engine, Docker Compose **2.24.4 or newer**,
and the Ubuntu system Python 3. The deployment script and HTTPS smoke test use only
Python's standard library; ML dependencies are entirely inside the container.
Create a deployment user and storage, running these commands as a server administrator:

```sh
sudo adduser --disabled-password --gecos '' deploy
sudo usermod -aG docker deploy
sudo install -d -o deploy -g deploy -m 0750 /opt/box-condition-api/releases /opt/box-condition-api/shared
sudo install -d -o deploy -g deploy -m 0700 /home/deploy/.ssh
```

Generate a dedicated CI SSH key on your own machine:

```sh
ssh-keygen -t ed25519 -f ~/.ssh/box-api-actions -C box-condition-api-actions -N ''
```

Add its `.pub` contents to `/home/deploy/.ssh/authorized_keys` on the server,
owned by `deploy` with mode 600. Verify a new SSH session can run `docker info`;
new group membership takes effect at login. Create
`/opt/box-condition-api/shared/.env`, owned by `deploy` with mode 600:

```dotenv
DEMO_DOMAIN=api.yourdomain.com
BOX_API_KEY=REPLACE_WITH_RANDOM_KEY_AT_LEAST_32_CHARACTERS
```

Replace the hostname and generate the actual key with `openssl rand -hex 32`.
Use literal `KEY=value` lines without shell expansion. The demo key stays on the
VPS and is read privately by the host-side smoke test; it is never put into the
release archive or GitHub repository. Give reviewers the key privately.

CI/CD SSH connections originate from GitHub-hosted runners. For this demo,
configure SSH for key authentication only and allow its port publicly in the
Hetzner firewall. A rule allowing only your personal IP blocks the workflow.
Verify both your administrative SSH key and the dedicated deployment key work
before disabling password authentication or changing firewall rules.

### GitHub configuration

After the first successful image publication, open the account's **Packages**
page, select `box-condition-api`, and set its visibility to **public**. GHCR
packages initially default to private; public packages can be pulled anonymously.
This matches the public repository and included model.

Create a GitHub environment named `production` without approval requirements.
Configure these environment secrets and variables:

| Kind | Name | Value |
| --- | --- | --- |
| Secret | DEPLOY_SSH_KEY | Contents of `~/.ssh/box-api-actions` |
| Secret | DEPLOY_KNOWN_HOSTS | Verified OpenSSH host-key entries for the VPS |
| Variable | DEPLOY_HOST | VPS IPv4 address or SSH hostname |
| Variable | DEPLOY_USER | `deploy` |
| Variable | DEPLOY_PORT | `22`, or your SSH port |
| Variable | DEMO_DOMAIN | Same hostname as the server `.env` |

To obtain the pinned host entry, run `ssh-keyscan -t ed25519 YOUR_VPS_IP`
(add `-p YOUR_PORT` for a custom port). Check its fingerprint against
`ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub` from the VPS console before
saving it as `DEPLOY_KNOWN_HOSTS`. The workflow requires strict host-key checking.

Finally, set **repository-level** variable `DEPLOY_ENABLED` to `true`; it must
be a repository variable because the job condition is evaluated before loading
the production environment. Leave it unset or `false` until setup is complete.
Push a new commit to `main`, or run **API CI/CD** manually from the Actions page.

### Release behavior and recovery

Each release lives in `/opt/box-condition-api/releases/<full-commit-sha>` with
its image digest, Compose/Caddy configuration, sample images, and smoke script.
The workflow validates configuration and pulls images before touching running
services, waits up to 300 seconds for readiness, and tests authenticated
concurrent predictions through HTTPS. It checks HTTPS readiness from the GitHub
runner too. The server's `current` symlink identifies the verified release;
`previous` retains the rollback release. Keep their images locally.

Failed readiness or predictions restore the previous image and configuration,
then verify recovery. Failed external HTTPS verification requests the same
rollback. A failed first deployment stops the API and preserves configuration
and certificate volumes. Rollback failures leave the workflow failed for investigation.

For manual rollback on the VPS:

```sh
python3 /opt/box-condition-api/current/deploy/release.py --rollback
```

For logs or a manual restart, select the current digest first:

```sh
export BOX_API_IMAGE=$(python3 -c 'import json; print(json.load(open("/opt/box-condition-api/current/release.json"))["image"])')
docker compose --env-file /opt/box-condition-api/shared/.env \
  -f /opt/box-condition-api/current/compose.yaml \
  -f /opt/box-condition-api/current/compose.production.yaml logs --tail 100
```

Use the same Compose options with `restart api` for a manual restart. A normal
shutdown drains workers for 30 seconds, then stops remaining worker processes
before deleting their temporary files. Failed model startup gets a separate
five-second termination budget, allowing the container to exit and restart.
Readiness failures alone do not restart a running container. Release queues
remain temporary and do not survive replacement or rollback.
