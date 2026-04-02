# Resource Shortlist (Science + Strong Subtitle Potential)

Use these as high-yield source pools, then enforce manual-subtitle filtering with `yt-dlp`.

## Topic Pools (TED)

- Biology: https://www.ted.com/topics/biology
- Medicine: https://www.ted.com/topics/medicine
- Chemistry: https://www.ted.com/topics/chemistry
- Physics: https://www.ted.com/topics/physics
- Geology: https://www.ted.com/topics/geology
- Climate science: https://www.ted.com/topics/climate+science

## YouTube Channels (Cross-domain + Science-heavy)

- TED: https://www.youtube.com/@TED/videos
- TEDxTalks: https://www.youtube.com/@TEDxTalks/videos
- MIT OpenCourseWare: https://www.youtube.com/@mitocw/videos
- World Science Festival (dialogue/panel friendly): https://www.youtube.com/@worldsciencefestival/videos
- Khan Academy: https://www.youtube.com/@khanacademy/videos
- Khan Academy Medicine: https://www.youtube.com/@KhanAcademyMedicine/videos
- HHMI BioInteractive: https://www.youtube.com/@HHMIBioInteractive/videos
- FermiLab: https://www.youtube.com/@fermilab/videos
- NASA: https://www.youtube.com/@NASA/videos
- USGS: https://www.youtube.com/@USGS/videos
- American Chemical Society: https://www.youtube.com/@AmericanChemicalSociety/videos

## Manual Subtitle Verification

For each candidate URL:

```bash
yt-dlp --dump-single-json --skip-download "<URL>" > meta.json
```

Then check:

- Keep if `meta.json` has non-empty `subtitles`.
- Skip if only `automatic_captions` exists.

Optional language check:

```bash
yt-dlp --list-subs "<URL>"
```
