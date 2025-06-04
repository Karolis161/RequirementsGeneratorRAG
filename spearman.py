import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class SpearmanCorrelationVisualizer:
    def __init__(self, filepath, evaluator_count=7):
        self.filepath = filepath
        self.evaluator_count = evaluator_count
        self.data = None
        self.header_row = None

    def load_data(self):
        df = pd.read_excel(self.filepath, header=None)
        self.header_row = df.iloc[0]
        self.data = df.iloc[1:self.evaluator_count + 1].reset_index(drop=True)

    def get_criterion_data(self, criterion_label):
        indices = [i for i, val in enumerate(self.header_row) if criterion_label in str(val)]
        criterion_df = self.data.iloc[:, indices]
        criterion_df.columns = [f"Req{i + 1}" for i in range(len(criterion_df.columns))]
        criterion_df = criterion_df.transpose()
        criterion_df.columns = [f"{i} Specialistas" for i in range(1, self.evaluator_count + 1)]
        return criterion_df.apply(pd.to_numeric)

    def compute_spearman(self, df):
        return df.corr(method='spearman')

    def plot_heatmap(self, corr_matrix, title):
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5)
        plt.title(title, fontsize=14)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    def run(self, criterion_label, plot_title=None):
        self.load_data()
        df = self.get_criterion_data(criterion_label)
        corr_matrix = self.compute_spearman(df)
        self.plot_heatmap(corr_matrix, plot_title or f"Spearman Correlation – {criterion_label}")
        return corr_matrix


visualizer = SpearmanCorrelationVisualizer("Requirement Quality Evaluation Form.xlsx")
corr_matrix = visualizer.run("[Lack of specificity]", "Spearman Koreliacija – Nepakankamas konkretumas")
