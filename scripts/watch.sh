#!/usr/bin/env bash
# Live status dashboard for a deciphaer-image-segmentation run.
#
# Sidecar to `pipeline.py --status`: clears the screen and reprints the status
# at a fixed interval. No alt-screen mode (so it can't leave the terminal in a
# weird state), no dependencies beyond bash + tput.
#
# Usage:
#   scripts/watch.sh configs/example_pipeline.yaml         # 10s interval (default)
#   scripts/watch.sh /path/to/run.yaml 5                  # 5s interval
#   Ctrl-C to exit.

set -u

CONFIG="${1:-}"
INTERVAL="${2:-10}"

if [[ -z "$CONFIG" ]]; then
    echo "Usage: $0 <config.yaml> [interval_seconds]" >&2
    exit 2
fi

if [[ ! -f "$CONFIG" ]]; then
    echo "Config not found: $CONFIG" >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Colors (tput probes terminfo; falls back to no-color if not a tty).
if [[ -t 1 ]]; then
    BOLD="$(tput bold)"
    DIM="$(tput dim)"
    RESET="$(tput sgr0)"
    WHITE="$(tput setaf 7)"
    CYAN="$(tput setaf 6)"
    BLUE="$(tput setaf 4)"
    GREEN="$(tput setaf 2)"
    YELLOW="$(tput setaf 3)"
    RED="$(tput setaf 1)"
else
    BOLD="" DIM="" RESET="" WHITE="" CYAN="" BLUE="" GREEN="" YELLOW="" RED=""
fi

print_header() {
    cat <<EOF
${BOLD}${WHITE}
:::::::::  :::::::::: :::::::: ::::::::::: :::::::::  :::    :::     :::     :::::::::: :::::::::
:+:    :+: :+:       :+:    :+:    :+:     :+:    :+: :+:    :+:   :+: :+:   :+:        :+:    :+:
+:+    +:+ +:+       +:+           +:+     +:+    +:+ +:+    +:+  +:+   +:+  +:+        +:+    +:+
+#+    +:+ +#++:++#  +#+           +#+     +#++:++#+  +#++:++#++ +#++:++#++: +#++:++#   +#++:++#:
+#+    +#+ +#+       +#+           +#+     +#+        +#+    +#+ +#+     +#+ +#+        +#+    +#+
#+#    #+# #+#       #+#    #+#    #+#     #+#        #+#    #+# #+#     #+# #+#        #+#    #+#
#########  ########## ######## ########### ###        ###    ### ###     ### ########## ###    ###
${RESET}
    ${WHITE}Multiomics Data Preprocessing Pipeline${RESET}
    ${DIM}Aldridge Lab${RESET}

    ${DIM}Config:${RESET}   ${CYAN}${CONFIG}${RESET}
    ${DIM}Refresh:${RESET}  ${WHITE}${INTERVAL}s${RESET}    ${DIM}(Ctrl-C to exit)${RESET}
EOF
}

# Color the raw `--status` output by matching state keywords. We pipe through
# sed; the output is otherwise plain text so any unmatched line passes through.
colorize_status() {
    sed -E \
        -e "s/(\[✓\])/${GREEN}\1${RESET}/g" \
        -e "s/(\[!\])/${RED}\1${RESET}/g" \
        -e "s/(\[·\])/${YELLOW}\1${RESET}/g" \
        -e "s/( done)([^a-zA-Z]|$)/${GREEN}\1${RESET}\2/g" \
        -e "s/(running[a-z ()/0-9]*)/${CYAN}${BOLD}\1${RESET}/g" \
        -e "s/(pending)/${DIM}\1${RESET}/g" \
        -e "s/(interrupted[a-z ()/0-9]*)/${RED}${BOLD}\1${RESET}/g" \
        -e "s/(DependencyNeverSatisfied)/${RED}${BOLD}\1${RESET}/gI" \
        -e "s/^(Pipeline:|Output:|MOMIA batches:|U-Net shards:|Downstream stages:|Skip list)/${BOLD}${BLUE}\1${RESET}/g"
}

# Hide the cursor + start with a fully clean screen + scrollback. We never
# leave more than the current frame in scrollback, so scrolling up only ever
# shows lines from the latest refresh.
trap 'tput cnorm 2>/dev/null; printf "\n"; exit 0' INT TERM EXIT
tput civis 2>/dev/null || true
printf '\033[H\033[2J\033[3J'

build_frame() {
    print_header
    echo
    if status="$(uv run python pipeline.py --config "$CONFIG" --status 2>/dev/null)"; then
        printf '%s\n' "$status" | colorize_status
    else
        printf '%s(failed to read status — pipeline.py exited non-zero)%s\n' "$RED" "$RESET"
    fi
    echo
    printf '%slast refresh: %s%s' "$DIM" "$(date '+%H:%M:%S')" "$RESET"
}

while :; do
    # 1. Build the whole frame off-screen first so the *previous* frame stays
    #    visible while pipeline.py --status is doing its slow startup.
    # 2. Atomic repaint: cursor home + clear visible + clear scrollback +
    #    write new frame, all in one printf. The terminal never sees a blank
    #    moment between clear and write, so there's no flicker.
    # 3. We do NOT truncate the frame: if it's taller than the window, the
    #    overflow lines naturally land in scrollback (and ONLY this frame's
    #    lines do, since we just wiped scrollback). Scroll up to see them.
    frame="$(build_frame)"
    printf '\033[H\033[2J\033[3J%s' "$frame"
    sleep "$INTERVAL"
done
