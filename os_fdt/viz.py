"""Static site generator for tournament runs (GitHub Pages compatible)."""

import json
import html
import shutil
from pathlib import Path


class StaticSiteGenerator:
    """Generates static HTML/CSS/JS files for tournament visualization."""

    def __init__(self, runs_dir: Path, output_dir: Path):
        """Initialize the generator.

        Args:
            runs_dir: Directory containing tournament run data
            output_dir: Directory to write generated static files (typically /docs)
        """
        self.runs_dir = runs_dir
        self.output_dir = output_dir
        self.css_content = self._get_css()

    def _get_css(self) -> str:
        """Get the CSS stylesheet content."""
        return """
/* Global Styles */
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    background: #f5f5f5;
    line-height: 1.6;
}

h1 { color: #333; margin-bottom: 10px; }
h2 { color: #444; margin-top: 0; }

/* Navigation */
.back-link {
    color: #0066cc;
    text-decoration: none;
    margin-bottom: 20px;
    display: inline-block;
}
.back-link:hover { text-decoration: underline; }

/* Index Page - Run List */
.run-list {
    background: white;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.run-item {
    padding: 15px;
    border-bottom: 1px solid #eee;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.run-item:last-child { border-bottom: none; }
.run-item:hover { background: #f9f9f9; }

.run-name {
    font-weight: 500;
    color: #0066cc;
    text-decoration: none;
    font-size: 16px;
}
.run-name:hover { text-decoration: underline; }

.run-meta {
    color: #666;
    font-size: 14px;
}

.empty-state {
    text-align: center;
    padding: 40px;
    color: #999;
}

/* Run Detail Page - Leaderboard */
.meta { color: #666; margin-bottom: 20px; }

.leaderboard, .rounds-section, .section {
    background: white;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}

table {
    width: 100%;
    border-collapse: collapse;
}

th {
    text-align: left;
    padding: 12px;
    background: #f9f9f9;
    border-bottom: 2px solid #ddd;
    font-weight: 600;
}

td {
    padding: 12px;
    border-bottom: 1px solid #eee;
}

tr:hover { background: #f9f9f9; }

.rank {
    font-weight: 600;
    color: #666;
    width: 60px;
}

.strategy { font-weight: 500; }

.score {
    text-align: right;
    font-weight: 600;
    color: #0066cc;
}

/* Rounds Section */
.round-link {
    display: inline-block;
    padding: 8px 12px;
    margin: 4px;
    background: #f0f0f0;
    border-radius: 4px;
    color: #0066cc;
    text-decoration: none;
}
.round-link:hover {
    background: #e0e0e0;
    text-decoration: underline;
}

.dictator-group {
    margin-bottom: 20px;
}

.rounds-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.round-link-detailed {
    display: inline-block;
    padding: 10px 14px;
    background: #f8f9fa;
    border: 1px solid #ddd;
    border-radius: 6px;
    color: #0066cc;
    text-decoration: none;
    min-width: 180px;
    transition: all 0.2s;
}

.round-link-detailed:hover {
    background: #e9ecef;
    border-color: #0066cc;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.round-number {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    color: #666;
    margin-bottom: 4px;
}

.round-players {
    font-size: 13px;
    color: #333;
    font-weight: 500;
}

/* Round Detail Page */
.participants {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 15px;
    margin-top: 10px;
}

.participant {
    padding: 12px;
    background: #f9f9f9;
    border-radius: 4px;
    border-left: 4px solid #ddd;
}

.participant.dictator {
    border-left-color: #0066cc;
    background: #f0f7ff;
}

.role {
    font-size: 12px;
    color: #666;
    text-transform: uppercase;
    font-weight: 600;
    margin-bottom: 4px;
}

.name {
    font-size: 16px;
    font-weight: 500;
}

.allocation {
    font-size: 18px;
    font-weight: 600;
    color: #0066cc;
    margin-top: 4px;
}

pre {
    background: #f5f5f5;
    padding: 15px;
    border-radius: 4px;
    overflow-x: auto;
    white-space: pre-wrap;
    word-wrap: break-word;
    font-size: 14px;
}

/* Tabs */
.tabs {
    display: flex;
    gap: 0;
    border-bottom: 1px solid #ddd;
    margin-bottom: 15px;
}

.tab {
    padding: 10px 20px;
    background: transparent;
    border: none;
    border-bottom: 3px solid transparent;
    cursor: pointer;
    font-size: 14px;
    font-weight: 500;
    color: #666;
    transition: all 0.2s;
}

.tab:hover {
    color: #0066cc;
    background: #f9f9f9;
}

.tab.active {
    color: #0066cc;
    border-bottom-color: #0066cc;
}

.tab-content {
    display: none;
}

.tab-content.active {
    display: block;
}

.markdown-content {
    line-height: 1.6;
}

.markdown-content h1,
.markdown-content h2,
.markdown-content h3,
.markdown-content h4,
.markdown-content h5,
.markdown-content h6 {
    margin-top: 1em;
    margin-bottom: 0.5em;
}

.markdown-content p {
    margin-bottom: 1em;
}

.markdown-content code {
    background: #f5f5f5;
    padding: 2px 6px;
    border-radius: 3px;
    font-family: 'Courier New', monospace;
    font-size: 13px;
}

.markdown-content pre {
    margin: 1em 0;
}

.markdown-content pre code {
    background: none;
    padding: 0;
}

.markdown-content ul,
.markdown-content ol {
    margin-bottom: 1em;
    padding-left: 2em;
}

.markdown-content blockquote {
    border-left: 4px solid #ddd;
    padding-left: 1em;
    margin: 1em 0;
    color: #666;
}

.markdown-content table {
    border-collapse: collapse;
    margin: 1em 0;
}

.markdown-content table th,
.markdown-content table td {
    border: 1px solid #ddd;
    padding: 8px 12px;
}

.markdown-content a {
    color: #0066cc;
    text-decoration: none;
}

.markdown-content a:hover {
    text-decoration: underline;
}

.nav-buttons {
    margin-top: 20px;
    display: flex;
    gap: 10px;
}

.nav-button {
    padding: 8px 16px;
    background: #0066cc;
    color: white;
    text-decoration: none;
    border-radius: 4px;
    display: inline-block;
}

.nav-button:hover {
    background: #0052a3;
}

.nav-button.disabled {
    background: #ccc;
    pointer-events: none;
    opacity: 0.6;
}
"""

    def _write_file(self, path: Path, content: str):
        """Write content to a file, creating directories as needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')

    def _load_roles(self, run_dir: Path) -> list[dict]:
        """Load role configuration from roles.json in run directory.

        Args:
            run_dir: Path to the tournament run directory

        Returns:
            List of role dictionaries with 'display', 'round_key', and 'allocation_key'.
            Falls back to default 3-player dictator game structure if roles.json doesn't exist.
        """
        roles_file = run_dir / 'roles.json'

        if roles_file.exists():
            try:
                with open(roles_file) as f:
                    roles = json.load(f)
                    # Validate that first role is dictator
                    if roles and roles[0]['round_key'] == 'dictator':
                        return roles
                    else:
                        print(f"Warning: First role in {roles_file} is not dictator, using defaults")
            except Exception as e:
                print(f"Warning: Error loading {roles_file}: {e}, using defaults")

        # Default to 3-player dictator game structure
        return [
            {"display": "Dictator", "round_key": "dictator", "allocation_key": "ME"},
            {"display": "Player B", "round_key": "player_b", "allocation_key": "B"},
            {"display": "Player C", "round_key": "player_c", "allocation_key": "C"}
        ]

    def _get_runs(self):
        """Get list of tournament runs with metadata."""
        runs = []

        if not self.runs_dir.exists():
            return runs

        for run_dir in sorted(self.runs_dir.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue

            run_info = {'name': run_dir.name}

            # Try to get metadata from leaderboard
            leaderboard_file = run_dir / 'leaderboard.json'
            if leaderboard_file.exists():
                try:
                    with open(leaderboard_file) as f:
                        leaderboard_data = json.load(f)
                        # Check for new format
                        if 'strategies' in leaderboard_data:
                            run_info['num_strategies'] = len(leaderboard_data['strategies'])
                        else:
                            # Old format
                            scores = leaderboard_data.get('scores', {})
                            run_info['num_strategies'] = len(scores)
                        run_info['total_rounds'] = leaderboard_data.get('total_rounds', 0)
                except:
                    pass

            # Count rounds from file if not in leaderboard
            if 'total_rounds' not in run_info or run_info['total_rounds'] == 0:
                rounds_file = run_dir / 'rounds.jsonl'
                if rounds_file.exists():
                    try:
                        with open(rounds_file) as f:
                            run_info['total_rounds'] = sum(1 for _ in f)
                    except:
                        pass

            runs.append(run_info)

        return runs

    def generate_index(self):
        """Generate the index.html page showing latest run with selector."""
        runs = self._get_runs()

        if not runs:
            # No runs available, show empty state
            html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tournament Results</title>
    <link rel="icon" type="image/svg+xml" href="resources/favicon.svg">
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="empty-state">No tournament runs found in runs/ directory</div>
</body>
</html>
"""
            self._write_file(self.output_dir / 'index.html', html_content)
            print(f"Generated index.html (empty state)")
            return

        # Get the latest run (first in the list since they're sorted reverse chronologically)
        latest_run = runs[0]
        latest_run_name = latest_run['name']

        # Load the latest run's content
        run_dir = self.runs_dir / latest_run_name
        leaderboard_file = run_dir / 'leaderboard.json'
        rounds_file = run_dir / 'rounds.jsonl'

        if not leaderboard_file.exists():
            print(f"Warning: No leaderboard found for latest run {latest_run_name}, creating simple index")
            self._generate_simple_index(runs)
            return

        # Load leaderboard data
        with open(leaderboard_file) as f:
            leaderboard_data = json.load(f)

        # Parse leaderboard
        if 'strategies' in leaderboard_data:
            strategies_data = leaderboard_data['strategies']
            leaderboard = [
                {
                    'strategy': name,
                    'total_score': data['total_score'],
                    'dictator_score': data.get('dictator_score', 0),
                    'player_score': data.get('player_score', 0),
                    'avg_dictator_score': data.get('avg_dictator_score', 0),
                    'avg_player_score': data.get('avg_player_score', 0),
                    'dictator_rounds': data.get('dictator_rounds', 0),
                    'player_rounds': data.get('player_rounds', 0),
                }
                for name, data in sorted(strategies_data.items(), key=lambda x: x[1]['total_score'], reverse=True)
            ]
        else:
            scores = leaderboard_data.get('scores', {})
            leaderboard = [
                {
                    'strategy': name,
                    'total_score': score,
                    'dictator_score': None,
                    'player_score': None,
                }
                for name, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
            ]

        num_rounds = leaderboard_data.get('total_rounds', 0)
        if num_rounds == 0 and rounds_file.exists():
            with open(rounds_file) as f:
                num_rounds = sum(1 for _ in f)

        # Generate the index page with embedded content
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tournament Results</title>
    <link rel="icon" type="image/svg+xml" href="resources/favicon.svg">
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div style="margin-bottom: 20px;">
        <label for="run-selector" style="font-weight: 500; margin-right: 10px;">Tournament Run:</label>
        <select id="run-selector" onchange="switchRun()" style="padding: 8px 12px; border-radius: 4px; border: 1px solid #ddd; font-size: 14px; min-width: 250px;">
"""

        # Add all runs to the dropdown
        for run in runs:
            run_name = run['name']
            run_display_name = run_name.replace('_', ' ')
            num_strategies = run.get('num_strategies', '?')
            run_rounds = run.get('total_rounds', '?')
            selected = 'selected' if run_name == latest_run_name else ''
            html_content += f"""            <option value="{html.escape(run_name)}" {selected}>{html.escape(run_display_name)} ({num_strategies} strategies, {run_rounds} rounds)</option>
"""

        html_content += f"""        </select>
    </div>

    <h1>{html.escape(latest_run_name.replace('_', ' '))}</h1>
    <div class="meta">{len(leaderboard)} strategies, {num_rounds} rounds</div>

    <div class="leaderboard">
        <h2>Leaderboard</h2>
        <table>
            <thead>
                <tr>
                    <th class="rank">Rank</th>
                    <th class="strategy">Strategy</th>
                    <th class="score">Total</th>"""

        has_breakdown = leaderboard and leaderboard[0].get('dictator_score') is not None

        if has_breakdown:
            html_content += """
                    <th class="score">As Dictator</th>
                    <th class="score">As Recipient</th>
                    <th class="score">Avg Dictator</th>
                    <th class="score">Avg Recipient</th>"""

        html_content += """
                </tr>
            </thead>
            <tbody>
"""

        for rank, entry in enumerate(leaderboard, 1):
            strategy = entry['strategy']
            total_score = entry['total_score']
            html_content += f"""                <tr>
                    <td class="rank">#{rank}</td>
                    <td class="strategy">{html.escape(strategy)}</td>
                    <td class="score">{total_score:.2f}</td>"""

            if has_breakdown:
                dictator_score = entry.get('dictator_score', 0)
                player_score = entry.get('player_score', 0)
                avg_dictator = entry.get('avg_dictator_score', 0)
                avg_player = entry.get('avg_player_score', 0)
                html_content += f"""
                    <td class="score">{dictator_score:.2f}</td>
                    <td class="score">{player_score:.2f}</td>
                    <td class="score">{avg_dictator:.2f}</td>
                    <td class="score">{avg_player:.2f}</td>"""

            html_content += """
                </tr>
"""

        html_content += """            </tbody>
        </table>
    </div>
"""

        # Add rounds section
        html_content += self._generate_rounds_section_html(
            latest_run_name,
            num_rounds,
            rounds_file,
            rounds_path_prefix=f"runs/{latest_run_name}/rounds/"
        )

        html_content += """
    <script>
        function switchRun() {
            const selector = document.getElementById('run-selector');
            const selectedRun = selector.value;
            window.location.href = 'runs/' + selectedRun + '/index.html';
        }
    </script>
</body>
</html>
"""

        self._write_file(self.output_dir / 'index.html', html_content)
        print(f"Generated index.html (showing latest run: {latest_run_name})")

    def _generate_rounds_section_html(self, run_name: str, num_rounds: int, rounds_file: Path, rounds_path_prefix: str = "rounds/", roles: list[dict] | None = None) -> str:
        """Generate the HTML for the rounds section.

        Args:
            run_name: Name of the run
            num_rounds: Total number of rounds
            rounds_file: Path to the rounds.jsonl file
            rounds_path_prefix: Prefix for round links (default: "rounds/")
            roles: List of role configurations (optional, will be loaded if not provided)

        Returns:
            HTML string for the rounds section
        """
        if num_rounds == 0:
            return ""

        # Load roles if not provided
        if roles is None:
            run_dir = self.runs_dir / run_name
            roles = self._load_roles(run_dir)

        html_content = ""

        # Load round data to organize by dictator
        rounds_by_dictator = {}
        all_strategies = set()

        if rounds_file.exists():
            with open(rounds_file) as f:
                for line in f:
                    round_data = json.loads(line)
                    idx = round_data['round']
                    dictator = round_data['dictator']

                    # Extract all participants dynamically from roles
                    participants = []
                    for role in roles:
                        participant_name = round_data.get(role['round_key'], '')
                        if participant_name:
                            all_strategies.add(participant_name)
                            if role['round_key'] != 'dictator':
                                participants.append(participant_name)

                    if dictator not in rounds_by_dictator:
                        rounds_by_dictator[dictator] = []

                    rounds_by_dictator[dictator].append({
                        'index': idx,
                        'participants': participants,  # Store all non-dictator participants
                    })

        # Generate rounds section with filter
        all_strategies_sorted = sorted(all_strategies)

        html_content += f"""    <div class="rounds-section">
        <h2>Rounds ({num_rounds} total)</h2>

        <div style="margin-bottom: 15px;">
            <label for="strategy-filter" style="font-weight: 500; margin-right: 10px;">Filter by strategy:</label>
            <select id="strategy-filter" onchange="filterRounds()" style="padding: 6px 12px; border-radius: 4px; border: 1px solid #ddd; font-size: 14px;">
                <option value="all">All rounds</option>
"""

        for strategy in all_strategies_sorted:
            html_content += f"""                <option value="{html.escape(strategy)}">{html.escape(strategy)}</option>
"""

        html_content += """            </select>
        </div>

        <div id="rounds-container">
"""

        # Organize rounds by dictator
        for dictator in sorted(rounds_by_dictator.keys()):
            rounds_list = rounds_by_dictator[dictator]
            html_content += f"""            <div class="dictator-group" data-dictator="{html.escape(dictator)}">
                <h3 style="color: #666; font-size: 16px; margin-top: 20px; margin-bottom: 10px;">Dictator: {html.escape(dictator)}</h3>
                <div class="rounds-grid">
"""

            for round_info in rounds_list:
                idx = round_info['index']
                participants_list = round_info['participants']
                # Build display string: "A vs B" or "A vs B vs C" etc.
                players_display = " vs ".join(participants_list)
                # Add data attributes for all participants for filtering
                data_attrs = f'data-players="{html.escape(" ".join(participants_list))}"'

                html_content += f"""                    <a href="{rounds_path_prefix}round-{idx}.html" class="round-link-detailed" {data_attrs}>
                        <div class="round-number">Round {idx}</div>
                        <div class="round-players">{html.escape(players_display)}</div>
                    </a>
"""

            html_content += """                </div>
            </div>
"""

        html_content += """        </div>
    </div>

    <script>
        function filterRounds() {
            const filter = document.getElementById('strategy-filter').value;
            const groups = document.querySelectorAll('.dictator-group');

            groups.forEach(group => {
                if (filter === 'all') {
                    group.style.display = 'block';
                    // Show all rounds in this group
                    group.querySelectorAll('.round-link-detailed').forEach(round => {
                        round.style.display = 'inline-block';
                    });
                } else {
                    const dictator = group.getAttribute('data-dictator');
                    const rounds = group.querySelectorAll('.round-link-detailed');
                    let hasVisibleRounds = false;

                    rounds.forEach(round => {
                        const players = round.getAttribute('data-players');
                        // Show round if filter matches dictator or any player
                        if (dictator === filter || players.includes(filter)) {
                            round.style.display = 'inline-block';
                            hasVisibleRounds = true;
                        } else {
                            round.style.display = 'none';
                        }
                    });

                    // Hide group if no rounds are visible
                    group.style.display = hasVisibleRounds ? 'block' : 'none';
                }
            });
        }
    </script>
"""

        return html_content

    def _generate_simple_index(self, runs):
        """Generate a simple index page with just a list of runs."""
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tournament Runs</title>
    <link rel="icon" type="image/svg+xml" href="resources/favicon.svg">
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <h1>Tournament Runs</h1>
    <div class="run-list">
"""

        for run in runs:
            run_name = run['name']
            run_display_name = run_name.replace('_', ' ')
            num_rounds = run.get('total_rounds', '?')
            num_strategies = run.get('num_strategies', '?')

            html_content += f"""        <div class="run-item">
            <a href="runs/{html.escape(run_name)}/index.html" class="run-name">{html.escape(run_display_name)}</a>
            <div class="run-meta">{num_strategies} strategies, {num_rounds} rounds</div>
        </div>
"""

        html_content += """    </div>
</body>
</html>
"""

        self._write_file(self.output_dir / 'index.html', html_content)
        print(f"Generated index.html (simple list)")

    def generate_run_page(self, run_name: str):
        """Generate the detail page for a specific run."""
        run_dir = self.runs_dir / run_name
        leaderboard_file = run_dir / 'leaderboard.json'
        rounds_file = run_dir / 'rounds.jsonl'

        if not leaderboard_file.exists():
            print(f"Warning: No leaderboard found for {run_name}, skipping")
            return

        # Load leaderboard
        with open(leaderboard_file) as f:
            leaderboard_data = json.load(f)

        # Check if we have the new detailed format
        if 'strategies' in leaderboard_data:
            # New format with detailed breakdown
            strategies_data = leaderboard_data['strategies']
            leaderboard = [
                {
                    'strategy': name,
                    'total_score': data['total_score'],
                    'dictator_score': data.get('dictator_score', 0),
                    'player_score': data.get('player_score', 0),
                    'avg_dictator_score': data.get('avg_dictator_score', 0),
                    'avg_player_score': data.get('avg_player_score', 0),
                    'dictator_rounds': data.get('dictator_rounds', 0),
                    'player_rounds': data.get('player_rounds', 0),
                }
                for name, data in sorted(strategies_data.items(), key=lambda x: x[1]['total_score'], reverse=True)
            ]
        else:
            # Old format with just scores
            scores = leaderboard_data.get('scores', {})
            leaderboard = [
                {
                    'strategy': name,
                    'total_score': score,
                    'dictator_score': None,
                    'player_score': None,
                }
                for name, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
            ]

        # Get round count
        num_rounds = leaderboard_data.get('total_rounds', 0)
        if num_rounds == 0 and rounds_file.exists():
            with open(rounds_file) as f:
                num_rounds = sum(1 for _ in f)

        run_display_name = run_name.replace('_', ' ')

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(run_display_name)} - Tournament Results</title>
    <link rel="icon" type="image/svg+xml" href="../../resources/favicon.svg">
    <link rel="stylesheet" href="../../styles.css">
</head>
<body>
    <a href="../../index.html" class="back-link">&larr; Back to all runs</a>
    <h1>{html.escape(run_display_name)}</h1>
    <div class="meta">{len(leaderboard)} strategies, {num_rounds} rounds</div>

    <div class="leaderboard">
        <h2>Leaderboard</h2>
        <table>
            <thead>
                <tr>
                    <th class="rank">Rank</th>
                    <th class="strategy">Strategy</th>
                    <th class="score">Total</th>"""

        # Check if we have detailed breakdown data
        has_breakdown = leaderboard and leaderboard[0].get('dictator_score') is not None

        if has_breakdown:
            html_content += """
                    <th class="score">As Dictator</th>
                    <th class="score">As Player</th>
                    <th class="score">Avg Dictator</th>
                    <th class="score">Avg Player</th>"""

        html_content += """
                </tr>
            </thead>
            <tbody>
"""

        for rank, entry in enumerate(leaderboard, 1):
            strategy = entry['strategy']
            total_score = entry['total_score']
            html_content += f"""                <tr>
                    <td class="rank">#{rank}</td>
                    <td class="strategy">{html.escape(strategy)}</td>
                    <td class="score">{total_score:.2f}</td>"""

            if has_breakdown:
                dictator_score = entry.get('dictator_score', 0)
                player_score = entry.get('player_score', 0)
                avg_dictator = entry.get('avg_dictator_score', 0)
                avg_player = entry.get('avg_player_score', 0)
                html_content += f"""
                    <td class="score">{dictator_score:.2f}</td>
                    <td class="score">{player_score:.2f}</td>
                    <td class="score">{avg_dictator:.2f}</td>
                    <td class="score">{avg_player:.2f}</td>"""

            html_content += """
                </tr>
"""

        html_content += """            </tbody>
        </table>
    </div>
"""

        # Add rounds section
        html_content += self._generate_rounds_section_html(run_name, num_rounds, rounds_file)

        html_content += """</body>
</html>
"""

        output_path = self.output_dir / 'runs' / run_name / 'index.html'
        self._write_file(output_path, html_content)
        print(f"Generated runs/{run_name}/index.html")

        return num_rounds

    def generate_round_pages(self, run_name: str, num_rounds: int):
        """Generate individual pages for each round."""
        run_dir = self.runs_dir / run_name
        rounds_file = run_dir / 'rounds.jsonl'

        if not rounds_file.exists():
            print(f"Warning: No rounds file found for {run_name}, skipping rounds")
            return

        # Load all rounds
        rounds_data = []
        with open(rounds_file) as f:
            for line in f:
                rounds_data.append(json.loads(line))

        # Generate a page for each round
        for round_data in rounds_data:
            self._generate_single_round_page(run_name, round_data, len(rounds_data))

    def _generate_single_round_page(self, run_name: str, round_data: dict, total_rounds: int):
        """Generate a single round detail page."""
        # Load roles configuration
        run_dir = self.runs_dir / run_name
        roles = self._load_roles(run_dir)

        # Extract round information
        round_index: int = round_data['round']
        result = round_data.get('allocation', {})
        prompt = round_data.get('prompt', '')
        response = round_data.get('response_text', '')

        run_display_name = run_name.replace('_', ' ')

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Round {round_index} - {html.escape(run_display_name)}</title>
    <link rel="icon" type="image/svg+xml" href="../../../resources/favicon.svg">
    <link rel="stylesheet" href="../../../styles.css">
    <script src="https://cdn.jsdelivr.net/npm/marked@11.1.1/marked.min.js"></script>
</head>
<body>
    <a href="../index.html" class="back-link">&larr; Back to {html.escape(run_display_name)}</a>
    <h1>Round {round_index}</h1>

    <div class="section">
        <h2>Participants</h2>
        <div class="participants">
"""

        # Generate participant cards dynamically from roles
        for i, role in enumerate(roles):
            participant_name = round_data.get(role['round_key'], '')
            allocation = result.get(role['allocation_key'], 0)
            css_class = "participant dictator" if i == 0 else "participant"

            html_content += f"""            <div class="{css_class}">
                <div class="role">{html.escape(role['display'])}</div>
                <div class="name">{html.escape(participant_name)}</div>
                <div class="allocation">{allocation} points</div>
            </div>
"""

        html_content += """        </div>
    </div>
"""

        # Generate unique IDs for this round's tabs
        prompt_id = f"prompt-{round_index}"
        response_id = f"response-{round_index}"

        if prompt:
            # Store the raw prompt in a data attribute for JavaScript access
            prompt_escaped = html.escape(prompt)
            html_content += f"""    <div class="section">
        <h2>Prompt</h2>
        <div class="tabs">
            <button class="tab active" onclick="switchTab('{prompt_id}', 'markdown')">Markdown</button>
            <button class="tab" onclick="switchTab('{prompt_id}', 'raw')">Raw</button>
        </div>
        <div id="{prompt_id}-markdown" class="tab-content markdown-content active">
            <div id="{prompt_id}-markdown-rendered"></div>
        </div>
        <div id="{prompt_id}-raw" class="tab-content">
            <pre>{prompt_escaped}</pre>
        </div>
        <script>
            // Render markdown for prompt
            document.getElementById('{prompt_id}-markdown-rendered').innerHTML = marked.parse({json.dumps(prompt)});
        </script>
    </div>
"""

        if response:
            # Store the raw response in a data attribute for JavaScript access
            response_escaped = html.escape(response)
            html_content += f"""    <div class="section">
        <h2>Response</h2>
        <div class="tabs">
            <button class="tab active" onclick="switchTab('{response_id}', 'markdown')">Markdown</button>
            <button class="tab" onclick="switchTab('{response_id}', 'raw')">Raw</button>
        </div>
        <div id="{response_id}-markdown" class="tab-content markdown-content active">
            <div id="{response_id}-markdown-rendered"></div>
        </div>
        <div id="{response_id}-raw" class="tab-content">
            <pre>{response_escaped}</pre>
        </div>
        <script>
            // Render markdown for response
            document.getElementById('{response_id}-markdown-rendered').innerHTML = marked.parse({json.dumps(response)});
        </script>
    </div>
"""

        # Navigation buttons
        html_content += """    <div class="nav-buttons">
"""

        if round_index > 1:
            html_content += f"""        <a href="round-{round_index - 1}.html" class="nav-button">&larr; Previous Round</a>
"""
        else:
            html_content += """        <span class="nav-button disabled">&larr; Previous Round</span>
"""

        if round_index < total_rounds:
            html_content += f"""        <a href="round-{round_index + 1}.html" class="nav-button">Next Round &rarr;</a>
"""
        else:
            html_content += """        <span class="nav-button disabled">Next Round &rarr;</span>
"""

        html_content += """    </div>

    <script>
        // Tab switching function
        function switchTab(sectionId, tabType) {
            // Get all tabs and contents for this section
            const rawTab = document.querySelector(`[onclick="switchTab('${sectionId}', 'raw')"]`);
            const markdownTab = document.querySelector(`[onclick="switchTab('${sectionId}', 'markdown')"]`);
            const rawContent = document.getElementById(`${sectionId}-raw`);
            const markdownContent = document.getElementById(`${sectionId}-markdown`);

            // Update active states
            if (tabType === 'raw') {
                rawTab.classList.add('active');
                markdownTab.classList.remove('active');
                rawContent.classList.add('active');
                markdownContent.classList.remove('active');
            } else {
                rawTab.classList.remove('active');
                markdownTab.classList.add('active');
                rawContent.classList.remove('active');
                markdownContent.classList.add('active');
            }
        }
    </script>
</body>
</html>
"""

        output_path = self.output_dir / 'runs' / run_name / 'rounds' / f'round-{round_index}.html'
        self._write_file(output_path, html_content)
        print(f"Generated runs/{run_name}/rounds/round-{round_index}.html")

    def generate_css(self):
        """Generate the styles.css file."""
        css_path = self.output_dir / 'styles.css'
        self._write_file(css_path, self.css_content)
        print("Generated styles.css")

    def generate_all(self):
        """Generate the complete static site."""
        print(f"Generating static site from {self.runs_dir} to {self.output_dir}")
        print()

        # Clean output directory while preserving resources
        if self.output_dir.exists():
            # Preserve the resources directory
            resources_dir = self.output_dir / 'resources'
            temp_resources = None
            temp_dir = None
            if resources_dir.exists():
                # Temporarily move resources to parent directory
                import tempfile
                temp_dir = Path(tempfile.mkdtemp())
                temp_resources = temp_dir / 'resources'
                shutil.copytree(resources_dir, temp_resources)

            # Remove the entire output directory
            shutil.rmtree(self.output_dir)
            self.output_dir.mkdir(parents=True)

            # Restore resources directory
            if temp_resources and temp_resources.exists() and temp_dir:
                shutil.copytree(temp_resources, resources_dir)
                shutil.rmtree(temp_dir)
        else:
            self.output_dir.mkdir(parents=True)

        # Generate CSS
        self.generate_css()
        print()

        # Generate index page
        self.generate_index()
        print()

        # Generate pages for each run
        runs = self._get_runs()
        for run in runs:
            run_name = run['name']
            print(f"Processing run: {run_name}")
            num_rounds = self.generate_run_page(run_name)
            if num_rounds and num_rounds > 0:
                self.generate_round_pages(run_name, num_rounds)
            print()

        print(f"✓ Static site generation complete!")
        print(f"✓ Output written to: {self.output_dir}")
        print(f"✓ Total runs: {len(runs)}")


def main():
    """Main entry point for static site generator."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate static HTML site for tournament visualization (GitHub Pages compatible)'
    )
    parser.add_argument(
        '--runs-dir',
        type=Path,
        default=None,
        help='Directory containing tournament runs (default: <repo>/runs)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory for generated site (default: <repo>/docs)'
    )

    args = parser.parse_args()

    # Determine repository root
    repo_root = Path(__file__).resolve().parent.parent

    # Set defaults relative to repo root
    runs_dir = args.runs_dir or repo_root / 'runs'
    output_dir = args.output_dir or repo_root / 'docs'

    # Validate runs directory exists
    if not runs_dir.exists():
        print(f"Error: Runs directory not found: {runs_dir}")
        return 1

    # Generate the site
    generator = StaticSiteGenerator(runs_dir, output_dir)
    generator.generate_all()

    return 0


if __name__ == '__main__':
    exit(main())
