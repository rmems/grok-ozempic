function strip_quotes(s,    out, i, c, in_s, in_d, in_b, esc) {
  out = ""
  in_s = 0; in_d = 0; in_b = 0
  for (i = 1; i <= length(s); i++) {
    c = substr(s, i, 1)
    if (esc) {
      if (in_s || in_d || in_b) out = out " "
      else out = out c
      esc = 0
    } else if (c == "\\") {
      esc = 1
    } else if (c == "'" && !in_d && !in_b) {
      in_s = !in_s
    } else if (c == "\"" && !in_s && !in_b) {
      in_d = !in_d
    } else if (c == "`" && !in_s) {
      in_b = !in_b
    } else if (!in_s && !in_d && !in_b) {
      out = out c
    } else {
      out = out " "
    }
  }
  gsub(/[ \t]+/, " ", out)
  return out
}
{
  line = strip_quotes($0)
  n = split(line, segs, /[;&|]+|&&|\|\|/)
  for (i = 1; i <= n; i++) {
    gsub(/^[ \t]+|[ \t]+$/, "", segs[i])
    if (segs[i] == "") continue
    m = split(segs[i], toks, /[ \t]+/)
    k = 1
    while (k <= m) {
      if (toks[k] == "command") {
        k++
      } else if (toks[k] == "env") {
        k++
        while (k <= m && (toks[k] ~ /^-/ || toks[k] ~ /^[A-Za-z_][A-Za-z0-9_]*=/)) {
          # env options that take a value: -u VAR, -C DIR
          if (toks[k] ~ /^-(u|C)$/) { k++; if (k > m) break }
          k++
        }
      } else if (toks[k] ~ /^[A-Za-z_][A-Za-z0-9_]*=/) {
        k++
      } else {
        break
      }
    }
    if (k <= m && toks[k] == "git") {
      j = k + 1
      global_help = 0
      while (j <= m && toks[j] ~ /^-/) {
        if (toks[j] == "--") {
          j++
          break
        }
        if (toks[j] ~ /^(--help|--version)$/ || toks[j] ~ /^-[hv]$/) {
          # git help/version with an optional topic; even if the topic is "commit",
          # no real commit is created.
          global_help = 1
        } else if (toks[j] ~ /^--(git-dir|work-tree|namespace|super-prefix|exec-path)=/ || toks[j] ~ /^--.+=/) {
          # value included in the option token
        } else if (toks[j] ~ /^-(C|c)$/ || toks[j] ~ /^--(git-dir|work-tree|namespace|super-prefix|exec-path)$/) {
          j++
          if (j > m) break
        } else if (toks[j] ~ /^--(html-path|man-path|info-path|paginate|no-pager|no-replace-objects|bare)$/ || toks[j] ~ /^-p$/) {
          # git no-value global options
        } else if (toks[j] ~ /^-[a-zA-Z]$/ || toks[j] ~ /^--[a-zA-Z-]+$/) {
          # other unknown short/long global options; treat as no-value
        }
        j++
      }
      if (j <= m && toks[j] == "commit" && !global_help) {
        # Reject commit-like operations that should not be attributed: --amend,
        # --dry-run, --help, -h. Stop scanning at the "--" path separator.
        for (p = j + 1; p <= m; p++) {
          if (toks[p] == "--") break
          if (toks[p] ~ /^(--amend|--dry-run|--help|-h)$/) { break }
        }
        if (p > m || toks[p] == "--") { exit 0 }
      }
    }
  }
  exit 1
}
