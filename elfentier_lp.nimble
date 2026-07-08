version       = "0.1.0"
author        = "Elfentier"
description   = "Elfentier.com waitlist landing (Crown 0.5.1 + Basolato 0.15.0)"
license       = "MIT"
srcDir        = "src"

requires "nim >= 2.2.0"
requires "crown >= 0.5.2"

task deps, "専用 NIMBLE_DIR に Crown 0.5.1 と依存をインストール":
  exec "bash scripts/bootstrap.sh"
