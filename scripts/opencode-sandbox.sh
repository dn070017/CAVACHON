#!/usr/bin/env bash
set -e

REPO_DIR="$(pwd)"
REPO_PARENT="$(dirname "$REPO_DIR")"

exec bwrap \
  --die-with-parent \
  --share-net \
  --proc /proc \
  --dev /dev \
  --tmpfs /tmp \
  \
  --ro-bind /usr /usr \
  --ro-bind /bin /bin \
  --ro-bind /lib /lib \
  --ro-bind /lib64 /lib64 \
  \
  --dir "$HOME" \
  --dir "$REPO_PARENT" \
  --dir /etc \
  \
  --bind "$HOME/.config/opencode" "$HOME/.config/opencode" \
  --bind "$HOME/.opencode" "$HOME/.opencode" \
  --bind "$HOME/.agents" "$HOME/.agents" \
  --bind "$HOME/.claude" "$HOME/.claude" \
  --bind "$HOME/.omo" "$HOME/.omo" \
  --ro-bind "$HOME/.bun" "$HOME/.bun" \
  --ro-bind "$HOME/.gitconfig" "$HOME/.gitconfig" \
  --bind "$HOME/.understand-anything" "$HOME/.understand-anything" \
  --ro-bind "$HOME/.understand-anything-plugin" "$HOME/.understand-anything-plugin" \
  --ro-bind "$HOME/.visual-explainer" "$HOME/.visual-explainer" \
  --bind "$HOME/.cc-safety-net" "$HOME/.cc-safety-net" \
  --bind "$HOME/.pixi" "$HOME/.pixi" \
  --bind "$HOME/.keras" "$HOME/.keras" \
  --ro-bind /etc/resolv.conf /etc/resolv.conf \
  --ro-bind /usr /usr \
  --ro-bind /bin /bin \
  --ro-bind /lib /lib \
  --ro-bind /lib64 /lib64 \
  --ro-bind /etc/resolv.conf /etc/resolv.conf \
  --ro-bind /etc/ssl /etc/ssl \
  \
  --setenv HOME "$HOME" \
  --setenv USER "$USER" \
  --setenv PATH "$HOME/.bun/bin:$PATH" \
  --setenv GITHUB_PERSONAL_ACCESS_TOKEN "$GITHUB_PERSONAL_ACCESS_TOKEN" \
  --setenv COMPOSIO_API_KEY "$COMPOSIO_API_KEY" \
  --setenv CONTEXT7_API_KEY "$CONTEXT7_API_KEY" \
  \
  --bind "$REPO_DIR" "$REPO_DIR" \
  --chdir "$REPO_DIR" \
  \
  opencode "$@"
