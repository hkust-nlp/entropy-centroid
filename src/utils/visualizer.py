"""
Visualization utilities for entropy analysis.
"""

import os
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class EntropyVisualizer:
    """
    Create visualizations for entropy analysis results.
    """

    def __init__(
        self,
        output_dir: str,
        dpi: int = 300,
        style: str = "seaborn-v0_8-darkgrid",
    ):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save visualizations
            dpi: DPI for static plots
            style: Matplotlib style
        """
        self.output_dir = output_dir
        self.dpi = dpi

        # Set matplotlib style
        try:
            plt.style.use(style)
        except Exception:
            # Fallback to default if style not available
            sns.set_theme()

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def plot_entropy_distribution(
        self,
        all_entropies: List[float],
        filename: str = "entropy_distribution",
    ):
        """
        Plot distribution of entropy values.

        Args:
            all_entropies: List of all entropy values
            filename: Output filename (without extension)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        ax1.hist(all_entropies, bins=50, edgecolor="black", alpha=0.7)
        ax1.set_xlabel("Entropy")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Entropy Distribution")
        ax1.grid(True, alpha=0.3)

        # Box plot
        ax2.boxplot(all_entropies, vert=True)
        ax2.set_ylabel("Entropy")
        ax2.set_title("Entropy Box Plot")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, f"{filename}.png"),
            dpi=self.dpi,
            bbox_inches="tight",
        )
        plt.close()

    def plot_entropy_over_position(
        self,
        results: List[Dict],
        max_samples: int = 10,
        filename: str = "entropy_over_position",
    ):
        """
        Plot entropy values over token positions for multiple samples.

        Args:
            results: List of generation results
            max_samples: Maximum number of samples to plot
            filename: Output filename (without extension)
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        for i, result in enumerate(results[:max_samples]):
            entropy_seq = result.get("entropy_sequence", [])
            positions = []
            entropies = []

            for item in entropy_seq:
                if item.get("entropy") is not None:
                    positions.append(item["position"])
                    entropies.append(item["entropy"])

            if positions:
                ax.plot(positions, entropies, alpha=0.6, label=f"Sample {i+1}")

        ax.set_xlabel("Token Position")
        ax.set_ylabel("Entropy")
        ax.set_title("Entropy Over Token Positions")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, f"{filename}.png"),
            dpi=self.dpi,
            bbox_inches="tight",
        )
        plt.close()

    def plot_high_entropy_tokens(
        self,
        high_entropy_freq: List[tuple],
        top_n: int = 20,
        filename: str = "high_entropy_tokens",
    ):
        """
        Plot most common high-entropy tokens.

        Args:
            high_entropy_freq: List of (token, count) tuples
            top_n: Number of top tokens to show
            filename: Output filename (without extension)
        """
        if not high_entropy_freq:
            return

        tokens, counts = zip(*high_entropy_freq[:top_n])

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.barh(range(len(tokens)), counts, alpha=0.7)
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens)
        ax.set_xlabel("Frequency")
        ax.set_title(f"Top {top_n} Most Common High-Entropy Tokens")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, f"{filename}.png"),
            dpi=self.dpi,
            bbox_inches="tight",
        )
        plt.close()

    def plot_statistics_summary(
        self,
        per_problem_stats: pd.DataFrame,
        filename: str = "statistics_summary",
    ):
        """
        Plot summary statistics for all problems.

        Args:
            per_problem_stats: DataFrame with per-problem statistics
            filename: Output filename (without extension)
        """
        if per_problem_stats.empty:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Average entropy per problem
        axes[0, 0].bar(range(len(per_problem_stats)), per_problem_stats["avg_entropy"])
        axes[0, 0].set_xlabel("Problem Index")
        axes[0, 0].set_ylabel("Average Entropy")
        axes[0, 0].set_title("Average Entropy per Problem")
        axes[0, 0].grid(True, alpha=0.3, axis="y")

        # Max entropy per problem
        axes[0, 1].bar(range(len(per_problem_stats)), per_problem_stats["max_entropy"], color="orange")
        axes[0, 1].set_xlabel("Problem Index")
        axes[0, 1].set_ylabel("Max Entropy")
        axes[0, 1].set_title("Maximum Entropy per Problem")
        axes[0, 1].grid(True, alpha=0.3, axis="y")

        # Token count per problem
        axes[1, 0].bar(range(len(per_problem_stats)), per_problem_stats["total_tokens"], color="green")
        axes[1, 0].set_xlabel("Problem Index")
        axes[1, 0].set_ylabel("Total Tokens")
        axes[1, 0].set_title("Total Tokens per Problem")
        axes[1, 0].grid(True, alpha=0.3, axis="y")

        # Std entropy per problem
        axes[1, 1].bar(range(len(per_problem_stats)), per_problem_stats["std_entropy"], color="red")
        axes[1, 1].set_xlabel("Problem Index")
        axes[1, 1].set_ylabel("Std Entropy")
        axes[1, 1].set_title("Entropy Standard Deviation per Problem")
        axes[1, 1].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, f"{filename}.png"),
            dpi=self.dpi,
            bbox_inches="tight",
        )
        plt.close()

    def create_interactive_entropy_plot(
        self,
        results: List[Dict],
        max_samples: int = 20,
        filename: str = "entropy_interactive",
    ):
        """
        Create interactive entropy visualization using Plotly.

        Args:
            results: List of generation results
            max_samples: Maximum number of samples to include
            filename: Output filename (without extension)
        """
        fig = go.Figure()

        for i, result in enumerate(results[:max_samples]):
            entropy_seq = result.get("entropy_sequence", [])
            positions = []
            entropies = []
            tokens = []

            for item in entropy_seq:
                if item.get("entropy") is not None:
                    positions.append(item["position"])
                    entropies.append(item["entropy"])
                    tokens.append(item["token"])

            if positions:
                fig.add_trace(
                    go.Scatter(
                        x=positions,
                        y=entropies,
                        mode="lines+markers",
                        name=f"Sample {result.get('id', i+1)}",
                        hovertemplate="<b>Position:</b> %{x}<br>"
                        + "<b>Entropy:</b> %{y:.3f}<br>"
                        + "<b>Token:</b> %{text}<extra></extra>",
                        text=tokens,
                    )
                )

        fig.update_layout(
            title="Token Entropy Over Positions (Interactive)",
            xaxis_title="Token Position",
            yaxis_title="Entropy",
            hovermode="closest",
            height=600,
        )

        fig.write_html(os.path.join(self.output_dir, f"{filename}.html"))

    def create_colored_token_html(
        self,
        results: List[Dict],
        max_samples: Optional[int] = None,
        filename: str = "colored_token_sequences",
    ):
        """
        Create HTML visualization with color-coded tokens based on entropy percentile.

        Color scheme:
        - Red: Top 1% highest entropy (most uncertain)
        - Pink: 1-3% entropy
        - Purple: 3-5% entropy
        - Black: Rest (normal)

        Args:
            results: List of generation results with entropy information
            max_samples: Maximum number of samples to display (None for all)
            filename: Output filename (without extension)
        """
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Token Entropy Visualization</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        h2 {
            color: #666;
            margin-top: 30px;
            border-left: 4px solid #2196F3;
            padding-left: 10px;
        }
        .legend {
            background-color: #f9f9f9;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            margin: 20px 0;
        }
        .legend-item {
            display: inline-block;
            margin-right: 20px;
            padding: 5px 10px;
        }
        .sample {
            margin: 30px 0;
            padding: 20px;
            background-color: #fafafa;
            border-radius: 5px;
            border: 1px solid #e0e0e0;
        }
        .sample-header {
            font-weight: bold;
            color: #444;
            margin-bottom: 15px;
            font-size: 1.1em;
        }
        .problem {
            background-color: #e3f2fd;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
            border-left: 4px solid #2196F3;
        }
        .tokens {
            line-height: 2.0;
            font-size: 1.1em;
            font-family: 'Courier New', monospace;
            padding: 15px;
            background-color: white;
            border-radius: 5px;
            word-wrap: break-word;
        }
        .token {
            cursor: help;
            padding: 2px 1px;
        }
        .token-red {
            color: #d32f2f;
            font-weight: bold;
            background-color: #ffebee;
        }
        .token-pink {
            color: #7b1fa2;
            font-weight: bold;
            background-color: #fce4ec;
        }
        .token-purple {
            color: #7cfc00;
            font-weight: bold;
            background-color: #f3e5f5;
        }
        .token-black {
            color: #333;
        }
        .stats {
            background-color: #fff3e0;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            font-size: 0.9em;
        }
        .tooltip {
            position: relative;
            display: inline-block;
        }
        .tooltip:hover::after {
            content: attr(data-entropy);
            position: absolute;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            background-color: #333;
            color: white;
            padding: 5px 10px;
            border-radius: 4px;
            white-space: nowrap;
            z-index: 1;
            font-size: 0.85em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎨 Token Entropy Visualization</h1>

        <div class="legend">
            <strong>Color Legend:</strong><br><br>
            <span class="legend-item token-red">■ Red: Top 1% (Highest Entropy)</span>
            <span class="legend-item token-pink">■ Pink: 1-3% Entropy</span>
            <span class="legend-item token-purple">■ Purple: 3-5% Entropy</span>
            <span class="legend-item token-black">■ Black: Normal</span>
            <br><br>
            <em>Hover over colored tokens to see their entropy values</em>
        </div>
"""

        # Limit samples if specified
        samples_to_show = results[:max_samples] if max_samples else results

        for i, result in enumerate(samples_to_show):
            sample_id = result.get("id", f"sample_{i+1}")
            problem = result.get("problem", "N/A")
            entropy_seq = result.get("entropy_sequence", [])
            stats = result.get("statistics", {})

            # Generate token HTML
            tokens_html = ""
            for item in entropy_seq:
                token = item.get("token", "")
                entropy = item.get("entropy")
                color = item.get("color", "black")
                percentile = item.get("percentile")

                # Escape HTML characters
                token = token.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

                # Create tooltip text
                if entropy is not None:
                    entropy_text = f"Entropy: {entropy:.3f}"
                    if percentile is not None:
                        entropy_text += f" | Percentile: {percentile:.1f}%"
                else:
                    entropy_text = "No entropy data"

                # Add token with appropriate color
                tokens_html += f'<span class="token token-{color} tooltip" data-entropy="{entropy_text}">{token}</span>'

            # Count high-entropy tokens
            high_entropy_counts = {
                "red": sum(1 for item in entropy_seq if item.get("color") == "red"),
                "pink": sum(1 for item in entropy_seq if item.get("color") == "pink"),
                "purple": sum(1 for item in entropy_seq if item.get("color") == "purple"),
            }

            # Create sample HTML
            html_content += f"""
        <div class="sample">
            <div class="sample-header">📊 Sample {i+1}: {sample_id}</div>

            <div class="problem">
                <strong>Problem:</strong> {problem[:200]}{"..." if len(problem) > 200 else ""}
            </div>

            <div class="stats">
                <strong>Statistics:</strong>
                Avg Entropy: {stats.get('avg_entropy', 0):.3f} |
                Max: {stats.get('max_entropy', 0):.3f} |
                Total Tokens: {stats.get('total_tokens', 0)} |
                High-Entropy: <span class="token-red">{high_entropy_counts['red']} red</span>,
                <span class="token-pink">{high_entropy_counts['pink']} purple</span>,
                <span class="token-purple">{high_entropy_counts['purple']} green</span>
            </div>

            <div class="tokens">
                {tokens_html}
            </div>
        </div>
"""

        html_content += """
    </div>
</body>
</html>
"""

        # Save HTML file
        output_path = os.path.join(self.output_dir, f"{filename}.html")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

    def create_all_visualizations(
        self,
        results: List[Dict],
        aggregate_stats: Dict,
        per_problem_stats: pd.DataFrame,
    ):
        """
        Create all visualizations.

        Args:
            results: List of generation results
            aggregate_stats: Aggregate statistics dictionary
            per_problem_stats: DataFrame with per-problem statistics
        """
        print("Creating visualizations...")

        # Collect all entropy values
        all_entropies = []
        for result in results:
            entropy_seq = result.get("entropy_sequence", [])
            for item in entropy_seq:
                if item.get("entropy") is not None:
                    all_entropies.append(item["entropy"])

        # Create plots
        if all_entropies:
            self.plot_entropy_distribution(all_entropies)
            print(f"  - Saved entropy distribution plot")

        if results:
            self.plot_entropy_over_position(results)
            print(f"  - Saved entropy over position plot")

            self.create_interactive_entropy_plot(results)
            print(f"  - Saved interactive entropy plot")

            # Create colored token visualization
            self.create_colored_token_html(results, max_samples=20)
            print(f"  - Saved colored token HTML visualization")

        if aggregate_stats.get("most_common_high_entropy_tokens"):
            self.plot_high_entropy_tokens(
                aggregate_stats["most_common_high_entropy_tokens"]
            )
            print(f"  - Saved high-entropy tokens plot")

        # Also plot colored high-entropy tokens
        if aggregate_stats.get("most_common_colored_high_entropy_tokens"):
            self.plot_high_entropy_tokens(
                aggregate_stats["most_common_colored_high_entropy_tokens"],
                filename="colored_high_entropy_tokens"
            )
            print(f"  - Saved colored high-entropy tokens plot")

        if not per_problem_stats.empty:
            self.plot_statistics_summary(per_problem_stats)
            print(f"  - Saved statistics summary plot")

        print(f"All visualizations saved to: {self.output_dir}")


def create_visualizer(config: Dict, output_dir: str) -> EntropyVisualizer:
    """
    Create a visualizer from configuration.

    Args:
        config: Configuration dictionary
        output_dir: Output directory for visualizations

    Returns:
        EntropyVisualizer instance
    """
    viz_config = config.get("output", {}).get("visualization", {})
    dpi = viz_config.get("dpi", 300)

    return EntropyVisualizer(output_dir=output_dir, dpi=dpi)
