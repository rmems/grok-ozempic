function remove_heredocs(s,    out, i, st, en, rest, marr, word, esc_word, redir_len, next_nl, body_start, rstart, rlen, pat) {
  out = ""
  i = 1
  while (i <= length(s)) {
    st = index(substr(s, i), "<<")
    if (st == 0) {
      out = out substr(s, i)
      break
    }
    st = i + st - 1
    rest = substr(s, st)
    # Match <<[-][']WORD["] and capture the delimiter word (alphanumeric/underscore
    # so we can safely embed it in a regex for the closing line).
    if (match(rest, /^<<-?[ \t]*[\"']?([A-Za-z_][A-Za-z0-9_]+)[\"']?/, marr)) {
      redir_len = RLENGTH
      word = marr[1]
      en = st + redir_len
      # The rest of the command line (if any) is kept so that a trailing
      # "; git commit" after the heredoc is still parsed.
      next_nl = index(substr(s, en), "\n")
      if (next_nl == 0) {
        # No body; not a real heredoc we can strip.
        out = out substr(s, i, st - i + 2)
        i = st + 2
        continue
      }
      body_start = en + next_nl
      # Closing delimiter line: optional leading whitespace/tabs (for <<-) then
      # the word, then optional trailing whitespace and a newline or end-of-string.
      esc_word = word
      gsub(/[.*+?|^$(){}[\]\\]/, "\\\\&", esc_word)
      pat = "\n[ \\t]*" esc_word "[ \\t]*(\n|$)"
      if (match(substr(s, body_start), pat, marr)) {
        rstart = RSTART
        rlen = RLENGTH
        # Keep everything up through the redirect token and the rest of the
        # first line, then jump past the heredoc body and its closing line.
        out = out substr(s, i, st - i)            # text before <<
        out = out substr(s, st, redir_len)        # the <<DELIM token
        if (next_nl > 1) out = out substr(s, en, next_nl - 1)  # after redirect on same line
        i = body_start + rstart + rlen - 1
        continue
      }
    }
    # Not a heredoc we can handle; keep the << and continue.
    out = out substr(s, i, st - i + 2)
    i = st + 2
  }
  return out
}

function strip_quotes(s,    out, i, c, in_s, in_d, in_b, esc) {
  out = ""
  in_s = 0; in_d = 0; in_b = 0
  for (i = 1; i <= length(s); i++) {
    c = substr(s, i, 1)
    if (esc) {
      esc = 0
    } else if (c == "\\") {
      esc = 1
    } else if (c == "'" && !in_d && !in_b) {
      if (in_s) {
        in_s = 0
        out = out "__Q__"
      } else {
        in_s = 1
      }
      continue
    } else if (c == "\"" && !in_s && !in_b) {
      if (in_d) {
        in_d = 0
        out = out "__Q__"
      } else {
        in_d = 1
      }
      continue
    } else if (c == "`" && !in_s) {
      if (in_b) {
        in_b = 0
        out = out "__Q__"
      } else {
        in_b = 1
      }
      continue
    }
    if (!in_s && !in_d && !in_b) out = out c
  }
  # Close any unterminated quoted region with a placeholder so the rest of
  # the command does not leak out as unquoted text.
  if (in_s || in_d || in_b) out = out "__Q__"
  gsub(/[ \t\n]+/, " ", out)
  return out
}

BEGIN {
  if (cmd == "") {
    while ((getline x) > 0) {
      if (line == "") line = x
      else line = line "\n" x
    }
  } else {
    line = cmd
  }
  line = remove_heredocs(line)
  line = strip_quotes(line)
  n = split(line, segs, /[;&|]+|&&|\|\|/)
  for (i = 1; i <= n; i++) {
    gsub(/^[ \t]+|[ \t]+$/, "", segs[i])
    if (segs[i] == "") continue
    m = split(segs[i], toks, /[ \t]+/)
    k = 1
    while (k <= m) {
      if (toks[k] == "command" || toks[k] ~ /^(sudo|nice|time|if|then|else|elif|do|while|until|for|\!|\()$/) {
        # Shell wrappers/control-flow keywords (or a subshell open paren) that
        # precede the actual command. This intentionally does not consume
        # option-argument pairs (e.g. "sudo -u x"); callers should write
        # "sudo -u x git commit" for now.
        k++
      } else if (toks[k] ~ /^\(/) {
        # A parenthesised command like (git commit) where the ( is attached to
        # the command word. Strip the leading paren and re-evaluate the token.
        toks[k] = substr(toks[k], 2)
        if (toks[k] == "") { k++; continue }
        # if the token also ends with ) (e.g. "(git)"), remove it too
        if (toks[k] ~ /\)$/) toks[k] = substr(toks[k], 1, length(toks[k]) - 1)
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
