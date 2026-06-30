# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__version__ = "1.8.37"

# Version of the downloadable data bundle (the heavy per-cohort percentile +
# representative shards). Bump when the DERIVED reference artifacts change — it pins
# the bundle filename, GitHub-release tag, and on-disk cache. Decoupled from
# __version__ so a code-only release reuses the last uploaded bundle. 5.23.0 made the
# shards dense in the canonical gene-ID space (oncoref#135 item 6). 5.23.3 is the
# first oncoref-owned QC-policy bundle: derived expression artifacts are rebuilt from
# source matrices with sample_qc="pass" by default and ship build/QC metadata.
DATA_VERSION = "5.23.3"

# Version of the per-cohort RAW source matrices (source_matrices.py). Independent of
# DATA_VERSION: the source matrices are the unchanging raw-TPM inputs, while DATA_VERSION
# tracks the derived bundle that's rebuilt from them. Canonicalization happens downstream
# (read/build time), so a canonical-space bundle bump must NOT repoint — or orphan the
# local caches of — these raw matrices. Bump only when a cohort's raw matrix changes.
SOURCE_MATRIX_VERSION = "5.22.6"

version_string = f"v{__version__}"


def print_version():
    print(version_string)


if __name__ == "__main__":
    print_version()
