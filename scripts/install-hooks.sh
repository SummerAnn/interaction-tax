#!/bin/sh
# Run once after cloning: sh scripts/install-hooks.sh

cat > .git/hooks/pre-commit << 'HOOK'
#!/bin/sh
PATTERNS="sk-or-v1-[a-f0-9]{60,} AIza[0-9A-Za-z_-]{35} sk-[a-zA-Z0-9]{48}"
for pattern in $PATTERNS; do
  match=$(git diff --cached --name-only | xargs grep -l -E "$pattern" 2>/dev/null)
  if [ -n "$match" ]; then
    echo "ERROR: Possible API key in staged files: $match"
    echo "Redact before committing."
    exit 1
  fi
done
exit 0
HOOK
chmod +x .git/hooks/pre-commit
echo "pre-commit hook installed"
