#!/bin/bash

# script-pre-commit.sh
FILES=()
while IFS= read -r FILE; do
    if [ -n "$FILE" ]; then
        FILES+=("$FILE")
    fi
done < <(git diff --cached --name-only --diff-filter=ACM)

if [ ${#FILES[@]} -ne 0 ]; then
    docker run -v "$(pwd)":/work -v "$(pwd)"/pre-commit/.cache:/home/myuser/.cache/pre-commit -w /work pre-commit-image pre-commit run --files "${FILES[@]}"
    if [ $? -eq 0 ]; then
        git add "${FILES[@]}"
    else
        echo "Pre-commit hooks failed"
        exit 1
    fi
fi
