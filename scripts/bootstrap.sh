#!/usr/bin/env bash
# 専用 NIMBLE_DIR に依存を隔離し、Crown CLI をインストールする。
# ``elfentier_lp.nimble`` の ``crown >= 0.5.1`` は nimble.directory の Git 参照を解決する（git が必要）。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export NIMBLE_DIR="${NIMBLE_DIR:-${ROOT}/.local_nimble}"
export PATH="${NIMBLE_DIR}/bin:${PATH}"
mkdir -p "${NIMBLE_DIR}"
tmp="$(mktemp -d)"
git clone --depth 1 --branch v0.15.0 https://github.com/itsumura-h/nim-basolato "${tmp}/basolato"
git clone --depth 1 --branch 0.5.2 https://github.com/nimmer-jp/crown "${tmp}/crown"
sed 's|requires "https://github.com/itsumura-h/nim-basolato#0.15.0"|requires "basolato == 0.15.0"|' \
  "${tmp}/crown/crown.nimble" > "${tmp}/crown/crown.nimble.tmp"
mv "${tmp}/crown/crown.nimble.tmp" "${tmp}/crown/crown.nimble"
(
  cd "${tmp}/basolato"
  nimble install -y
)
(
  cd "${tmp}/crown"
  nimble install -y
)
nimble install -y
echo "bootstrap: NIMBLE_DIR=${NIMBLE_DIR}"
