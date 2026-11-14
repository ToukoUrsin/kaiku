#!/usr/bin/env bash
# Put the current tree on the hosted demo. The Train button's functions on Modal are deployed
# first, from the same tree: a policy trained there is loaded here, so the two must match.
#
#   deploy/deploy.sh            needs hsec FLY_API_TOKEN and JUNCTION_MODAL_TOKEN_ID / _SECRET
set -euo pipefail
cd "$(dirname "$0")/.."
hsec exec --only JUNCTION_MODAL_TOKEN_ID,JUNCTION_MODAL_TOKEN_SECRET -- bash -c \
  'export MODAL_TOKEN_ID=$JUNCTION_MODAL_TOKEN_ID MODAL_TOKEN_SECRET=$JUNCTION_MODAL_TOKEN_SECRET; modal deploy tools/modal_policy.py' | tail -2
hsec exec --only FLY_API_TOKEN,JUNCTION_MODAL_TOKEN_ID,JUNCTION_MODAL_TOKEN_SECRET -- bash -c \
  'fly secrets set --app kaiku-echo --stage MODAL_TOKEN_ID="$JUNCTION_MODAL_TOKEN_ID" MODAL_TOKEN_SECRET="$JUNCTION_MODAL_TOKEN_SECRET" >/dev/null'
hsec exec --only FLY_API_TOKEN -- fly deploy --config deploy/fly/fly.toml --dockerfile deploy/fly/Dockerfile --ha=false --yes | tail -3
for f in index.html js/hud.js; do
  [ "$(curl -s "https://kaiku-echo.fly.dev/$f" | sha1sum)" = "$(sha1sum < "kaiku/web/static/$f")" ] && echo "$f is live" || echo "$f on the site is NOT this tree's"
done
