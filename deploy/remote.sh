#!/usr/bin/env bash
# GitHub runner entry point. Credentials arrive through masked environment secrets.
set -euo pipefail
: "${DEPLOY_SSH_KEY:?Missing deployment key}"
: "${DEPLOY_KNOWN_HOSTS:?Missing pinned SSH host keys}"
: "${DEPLOY_HOST:?Missing server hostname}"
: "${DEPLOY_USER:?Missing deployment user}"
: "${DEMO_DOMAIN:?Missing demo hostname}"
: "${GITHUB_SHA:?Missing release SHA}"
: "${BOX_API_IMAGE:?Missing tested image digest}"
DEPLOY_PORT=${DEPLOY_PORT:-22}
[[ "$DEPLOY_HOST" =~ ^[a-zA-Z0-9.-]+$ ]]
[[ "$DEPLOY_USER" =~ ^[a-z_][a-z0-9_-]*$ ]]
[[ "$DEPLOY_PORT" =~ ^[0-9]{1,5}$ ]]
[[ "$DEMO_DOMAIN" =~ ^[a-zA-Z0-9.-]+$ ]]
[[ "$GITHUB_SHA" =~ ^[0-9a-f]{40}$ ]]
[[ "$BOX_API_IMAGE" =~ ^ghcr.io/sepehringo/box-condition-api@sha256:[0-9a-f]{64}$ ]]
port=$((10#$DEPLOY_PORT))
(( port > 0 && port < 65536 ))

ssh_dir=$(mktemp -d)
trap 'rm -rf "$ssh_dir"' EXIT
printf '%s\n' "$DEPLOY_SSH_KEY" > "$ssh_dir/key"
printf '%s\n' "$DEPLOY_KNOWN_HOSTS" > "$ssh_dir/known_hosts"
chmod 600 "$ssh_dir/key" "$ssh_dir/known_hosts"
options=(-i "$ssh_dir/key" -o IdentitiesOnly=yes -o BatchMode=yes
         -o StrictHostKeyChecking=yes -o "UserKnownHostsFile=$ssh_dir/known_hosts"
         -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=3)
destination="$DEPLOY_USER@$DEPLOY_HOST"
release="/opt/box-condition-api/releases/$GITHUB_SHA"
ssh -p "$port" "${options[@]}" "$destination" "mkdir -p '$release'"
scp -P "$port" "${options[@]}" .deployment-artifacts/release.tar.gz "$destination:$release/release.tar.gz"
ssh -p "$port" "${options[@]}" "$destination" \
    "tar -xzf '$release/release.tar.gz' -C '$release' && python3 '$release/deploy/release.py' --release '$GITHUB_SHA' --image '$BOX_API_IMAGE' --domain '$DEMO_DOMAIN'"

# Confirm reachability from outside the VPS as well as the host-side API smoke test.
if ! curl --fail --silent --show-error --retry 5 --retry-all-errors --retry-delay 5 \
    --max-time 10 "https://$DEMO_DOMAIN/ready"; then
    printf '\nExternal HTTPS verification failed; requesting rollback.\n' >&2
    ssh -p "$port" "${options[@]}" "$destination" \
        "python3 '$release/deploy/release.py' --rollback --release '$GITHUB_SHA' --domain '$DEMO_DOMAIN'"
    exit 1
fi
