# Third-Party Notices

BGM Montage is distributed under AGPL-3.0-or-later. The components below are
direct runtime or development dependencies declared by this repository. Their
licenses remain separate from BGM Montage; transitive dependencies may carry
additional notices and must be reviewed from the installed distribution before
redistribution.

| Component | Purpose | License | Upstream |
| --- | --- | --- | --- |
| NumPy | Numerical arrays | BSD-3-Clause (project) | <https://github.com/numpy/numpy> |
| SciPy | Signal processing | BSD-3-Clause (project) | <https://github.com/scipy/scipy> |
| librosa | Audio feature extraction | ISC | <https://github.com/librosa/librosa> |
| SoundFile | Audio decoding | BSD-3-Clause | <https://github.com/bastibe/python-soundfile> |
| OpenCV | Image/video analysis | Apache-2.0 | <https://github.com/opencv/opencv> |
| Requests | HTTP acquisition | Apache-2.0 | <https://github.com/psf/requests> |
| python-dotenv | Environment loading | BSD-3-Clause | <https://github.com/theskumar/python-dotenv> |
| Pillow | Image processing | MIT-CMU | <https://github.com/python-pillow/Pillow/blob/main/LICENSE> |
| ImageHash | Perceptual hashing | BSD-2-Clause | <https://pypi.org/project/ImageHash/4.3.2/> |
| PyYAML | YAML configuration | MIT | <https://github.com/yaml/pyyaml> |
| PyTorch | Optional semantic analysis | BSD-3-Clause | <https://github.com/pytorch/pytorch> |
| torchvision | Optional vision transforms | BSD-3-Clause | <https://github.com/pytorch/vision> |
| Transformers | Optional CLIP/semantic models | Apache-2.0 | <https://github.com/huggingface/transformers> |
| pytest | Test runner | MIT | <https://github.com/pytest-dev/pytest> |
| yt-dlp | Optional YouTube acquisition | Unlicense | <https://github.com/yt-dlp/yt-dlp> |

The pinned `requirements.lock.txt` metadata was also reviewed. Notable
transitive license expressions include MPL-2.0 (`certifi`), LGPL-2.1-or-later
(`soxr`), PSF-2.0 (`typing_extensions`), BSD/Apache with the LLVM exception
(`llvmlite`), and Apache-2.0/CNRI-Python (`regex`). These remain separate
dependencies; retain each installed distribution's license files if shipping
a bundled environment. NumPy wheels include additional 0BSD, MIT, Zlib, and
CC0 notices. SciPy distributions may include OpenBLAS, LAPACK, GCC runtime,
or libquadmath components with their own BSD, GPL-with-GCC-exception, or LGPL
notices, depending on the platform and wheel build. The exact installed
wheel's `.dist-info/licenses` files are authoritative for redistribution.

No incompatibility was identified in the pinned Python dependency metadata
for relicensing this repository's own code under AGPL-3.0-or-later. This is a
provenance review, not legal advice; bundled binary distributions still need
their own notice review.

The repository does not bundle stock footage, music, model weights, fonts, or
other third-party media. Downloaded or user-provided media is external to this
repository. Before publication, record the source URL, license, attribution,
and any usage restrictions in the generated asset manifest and comply with the
source provider's current terms.

FFmpeg is invoked as an external executable and is not redistributed by this
repository. Its own build and codec licenses apply to the installed binary and
the selected encoding configuration.
