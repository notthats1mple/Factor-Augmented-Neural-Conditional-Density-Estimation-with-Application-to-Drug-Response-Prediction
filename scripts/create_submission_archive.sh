#!/usr/bin/env bash
set -euo pipefail

archive_name="${1:-cindes_statsmed_submission_code_results.tar.gz}"

repo_root="$(git rev-parse --show-toplevel)"
build_root="$(mktemp -d "${TMPDIR:-/tmp}/cindes_deposit.XXXXXX")"
deposit_dir="$build_root/cindes_statsmed_submission_code_results"

cleanup() {
  rm -rf "$build_root"
}
trap cleanup EXIT

mkdir -p "$deposit_dir"

copy_file() {
  local src="$1"
  local dest="$2"
  mkdir -p "$(dirname "$dest")"
  cp "$repo_root/$src" "$dest"
}

cd "$repo_root"

while IFS= read -r file; do
  case "$file" in
    fac_cindes/results/*|fac_cindes/figures/*|fac_cindes/tables/*)
      ;;
    fac_cindes/*)
      copy_file "$file" "$deposit_dir/code/$file"
      ;;
    real_app_gdsc/*.py|real_app_gdsc/README.md|real_app_gdsc/data/raw/.gitkeep|real_app_gdsc/data/processed/.gitkeep)
      copy_file "$file" "$deposit_dir/code/$file"
      ;;
    real_app_gdsc/results/*)
      copy_file "$file" "$deposit_dir/results/real_app_gdsc/${file#real_app_gdsc/results/}"
      ;;
    real_app_gdsc/figures/*)
      copy_file "$file" "$deposit_dir/figures/real_app_gdsc/${file#real_app_gdsc/figures/}"
      ;;
    tables/*)
      copy_file "$file" "$deposit_dir/$file"
      ;;
    reports/*)
      copy_file "$file" "$deposit_dir/$file"
      ;;
    README.md|CODE_AVAILABILITY.md|CITATION.cff|LICENSE|requirements.txt|.gitignore|REPOSITORY_DEPOSIT_GUIDE.md)
      copy_file "$file" "$deposit_dir/$file"
      ;;
  esac
done < <(git ls-files)

for optional_file in \
  factor_augmented_cindes_paper.tex \
  supporting_information.tex \
  README_TARGETED_REVISION_4ITEMS.md
do
  if [[ -f "$optional_file" ]]; then
    copy_file "$optional_file" "$deposit_dir/$optional_file"
  fi
done

COPYFILE_DISABLE=1 tar \
  --exclude='._*' \
  --exclude='.DS_Store' \
  -czf "$repo_root/$archive_name" \
  -C "$build_root" \
  "$(basename "$deposit_dir")"

if tar -tzf "$repo_root/$archive_name" | grep -E '(^|/)(\._|\.DS_Store)' >/dev/null; then
  echo "Archive contains macOS metadata files; refusing to finish." >&2
  exit 1
fi

echo "Created $repo_root/$archive_name"
