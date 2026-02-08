# Tournament Visualization Site

This directory contains the static HTML/CSS files for visualizing tournament runs.

## Generating the Site

To generate or update the static site, run:

```bash
python -m os_fdt.viz
```

This will:
- Read tournament data from `/runs` directory
- Generate static HTML/CSS files in `/docs` directory
- Create an index page listing all tournament runs
- Create detail pages for each run with leaderboards
- Create individual pages for each round with full details

## GitHub Pages Setup

To publish this site on GitHub Pages:

1. Commit the generated `/docs` directory to your repository
2. Go to your repository settings on GitHub
3. Navigate to Pages section
4. Under "Source", select "Deploy from a branch"
5. Select the branch (e.g., `main` or `master`)
6. Select `/docs` as the folder
7. Click Save

Your site will be available at: `https://michaelrglass.github.io/os-fdt/`

## Local Preview

To preview the site locally, you can use Python's built-in HTTP server:

```bash
cd docs
python -m http.server 8000
```

Then open `http://localhost:8000` in your browser.

## Site Structure

```
docs/
├── index.html              # List of all tournament runs
├── styles.css              # Shared stylesheet
└── runs/
    └── <run-name>/
        ├── index.html      # Run detail page with leaderboard
        └── rounds/
            └── round-*.html  # Individual round detail pages
```

## Customization

To customize the appearance, edit the CSS in `os_fdt/viz.py` in the `_get_css()` method, then regenerate the site.