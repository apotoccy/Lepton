# Development transcripts

This directory contains the 52 `.docx` transcripts (~43 MB) from the
AI-assisted development sessions that produced Lepton. They are useful history
and rationale, but they are **not build inputs** and **not a current
specification**.

The transcripts are excluded from the default Git clone by `../.gitignore` to
keep the repository small and clone-friendly. The current documentation is
`../README.md` and `../ROADMAP.md`.

## If you want them in version control

Use Git LFS, because these are binary `.docx` files:

```bash
cd ..                       # repository root
git lfs install
git lfs track "READFILES/*.docx"
git add .gitattributes
git add -f READFILES/*.docx
git commit -m "Add development transcripts via Git LFS"
git push
```

Alternatively, `git add -f READFILES/*.docx` adds them as ordinary Git blobs,
but every clone will download the full ~43 MB.

Check GitHub's Git LFS storage and bandwidth quotas before doing this.
