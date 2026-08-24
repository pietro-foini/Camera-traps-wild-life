#!/bin/bash

# Configuration and source paths for ml-invoice
CONFIG_FILE="mypy.ini"

# Run mypy and save the output to the "output" variable
output=$(mypy --show-error-codes --no-pretty --no-error-summary --config-file $CONFIG_FILE . 2>&1)

# Redirect the stdout to a file named "mypy_recap.txt"
echo "--- Mypy Run: $(date) ---" > mypy_recap.txt
echo "$output" | grep -E 'error:|: note:|: warning:' | grep -v ' error: \(this|TypeVar\)' | grep -v 'warning: unused import' >> mypy_recap.txt

# Print output to console as well
echo "$output"

exit 0