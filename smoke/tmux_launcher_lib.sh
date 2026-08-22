#!/usr/bin/env bash
# Shared fail-closed helpers for project tmux launchers.

# Do not use TMUX as an application variable. tmux exports that name to
# describe the current client connection, and overwriting it can make the
# server treat an executable path as its socket path.

select_tmux_binary() {
  local candidate
  for candidate in /usr/bin/tmux /bin/tmux; do
    if [[ -f "$candidate" && -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  candidate="$(type -P tmux 2>/dev/null || true)"
  if [[ -n "$candidate" && -f "$candidate" && -x "$candidate" ]]; then
    printf '%s\n' "$candidate"
    return 0
  fi
  if [[ "${1:-}" != "--quiet" ]]; then
    echo "ERROR: no executable system tmux binary is available" >&2
  fi
  return 1
}

tmux_job_server() {
  if [[ -z "${TMUX_BIN_PATH:-}" || ! -f "$TMUX_BIN_PATH" || ! -x "$TMUX_BIN_PATH" ]]; then
    echo "ERROR: TMUX_BIN_PATH is not an executable regular file" >&2
    return 1
  fi
  if [[ -z "${TMUX_SERVER_LABEL:-}" ]]; then
    echo "ERROR: TMUX_SERVER_LABEL is unset" >&2
    return 1
  fi
  env -u TMUX -u TMUX_PANE TMUX_TMPDIR=/tmp \
    "$TMUX_BIN_PATH" -L "$TMUX_SERVER_LABEL" "$@"
}

tmux_attach_hint() {
  local session="$1"
  printf 'env -u TMUX -u TMUX_PANE TMUX_TMPDIR=/tmp %q -L %q attach -t %q' \
    "$TMUX_BIN_PATH" "$TMUX_SERVER_LABEL" "$session"
}

tmux_attach_or_report() {
  local session="$1"
  if [[ -n "${TMUX:-}" ]]; then
    echo "Already inside tmux; the job remains safely detached."
    printf 'Attach from a separate terminal: '
    tmux_attach_hint "$session"
    printf '\n'
    return 0
  fi
  tmux_job_server attach -t "$session"
}
