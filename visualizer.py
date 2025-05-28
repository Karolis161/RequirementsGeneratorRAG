import matplotlib.pyplot as plt


def plot_rag_vs_no_rag(results_rag, results_no_rag):
    labels = list(results_rag.keys())
    rag_values = [results_rag[k] for k in labels]
    no_rag_values = [results_no_rag[k] for k in labels]

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_no_rag = ax.bar([i - width / 2 for i in x], no_rag_values, width, label='No RAG', color='gray')
    bars_rag = ax.bar([i + width / 2 for i in x], rag_values, width, label='RAG', color='skyblue')

    ax.set_ylabel('Balas')
    ax.set_title('RAG vs No-RAG Metrics Comparison')
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout()

    def add_labels(bars, color):
        for bar in bars:
            height = bar.get_height()
            xpos = bar.get_x() + bar.get_width() / 2
            label = f"{height:.3f}"
            max_height = max(max(rag_values), max(no_rag_values))
            ax.set_ylim(0, max(1.15, max_height + 0.1))

            ypos = height + 0.015 if height < 0.1 else height + 0.01

            ax.text(
                xpos,
                ypos,
                label,
                ha='center',
                va='bottom',
                fontsize=9,
                color=color,
                clip_on=False,
                zorder=10
            )

    add_labels(bars_no_rag, 'white')
    add_labels(bars_rag, 'black')

    plt.savefig("rag_vs_no_rag_results.png")
    plt.show()


def plot_comparison(data, title, filename, ylabel="Balas"):
    labels = list(data.keys())
    systems = list(next(iter(data.values())).keys())

    x = range(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['gray', 'skyblue', 'steelblue']
    bars = []
    for i, system in enumerate(systems):
        values = [data[label][system] for label in labels]
        offset = -width + i * width
        bars.append(ax.bar([xi + offset for xi in x], values, width, label=system, color=colors[i]))

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout()

    for bar_group in bars:
        for bar in bar_group:
            height = bar.get_height()
            xpos = bar.get_x() + bar.get_width() / 2
            label = f"{height:.2f}"
            ypos = height + 0.015 if height < 0.1 else height + 0.01
            ax.text(xpos, ypos, label, ha='center', va='bottom', fontsize=9)

    plt.savefig(filename)
    plt.show()


metric_results = {
    "Answer Relevance": {
        "Be RAG": 0.058,
        "Įprastas RAG": 0.36,
        "Išsamus RAG": 0.48,
    },
    "Faithfulness": {
        "Be RAG": 0.0,
        "Įprastas RAG": 0.1,
        "Išsamus RAG": 0.4,
    },
    "Recall@k": {
        "Be RAG": 0.0,
        "Įprastas RAG": 0.2,
        "Išsamus RAG": 0.4,
    },
    "Technical Term Coverage": {
        "Be RAG": 0.107,
        "Įprastas RAG": 0.1,
        "Išsamus RAG": 0.03,
    },
}

quality_results = {
    "Neapibrėžtumas": {
        "Be RAG": 0.31,
        "Įprastas RAG": 0.32,
        "Išsamus RAG": 0.19,
    },
    "Nepakankamas konkretumas": {
        "Be RAG": 0.45,
        "Įprastas RAG": 0.38,
        "Išsamus RAG": 0.28,
    },
    "Daugialypumas": {
        "Be RAG": 0.68,
        "Įprastas RAG": 0.31,
        "Išsamus RAG": 0.25,
    },
}

plot_comparison(metric_results, "RAG sistemų palyginimas pagal metrikas", "metrics_comparison.png")
plot_comparison(quality_results, "RAG sistemų palyginimas pagal kokybės kriterijus", "quality_comparison.png")
