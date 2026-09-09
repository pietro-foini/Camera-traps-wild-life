#!/bin/bash

# Function to check if a given tag is already present in the Git repository
is_tag_present() {
  local given_tag=$1
  git fetch --tags &> /dev/null
  local tags=$(git tag -l)

  for tag in $tags; do
    if [ "$tag" == "$given_tag" ]; then
      return 0
    fi
  done

  return 1
}

ACTUAL_VERSION=$(poetry version -s)

# Get the target branch from the first argument or GitHub Actions environment variable
TARGET_BRANCH=${1:-$GITHUB_REF_NAME}
echo "${TARGET_BRANCH^^} VERSION: $ACTUAL_VERSION"

if [[ $TARGET_BRANCH == "master" || $TARGET_BRANCH == "main" ]]; then
  if is_tag_present $ACTUAL_VERSION; then
    echo "Tag ${ACTUAL_VERSION} is already present in the repository. Please change it."
    exit 1
  fi
  if [[ $2 == "true" ]]; then
    echo "Tag $ACTUAL_VERSION is not present in the repository. Proceeding to add it"
    git tag -a "${ACTUAL_VERSION}" -m "version ${ACTUAL_VERSION}"
    git push origin "${ACTUAL_VERSION}"
  fi
fi