import NewsModel
import EmmaModel
import io
import collections
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
class similarityComparison:
    def __init__(self):
        self.result = defaultdict(dict)

        human_comp = pd.read_csv("wordsim353/combined.csv")
        human_comp = human_comp[human_comp["Human (mean)"] >= 5.0]
        human_comp["Human (mean)"] /= 10
        human_comp = human_comp[human_comp['Word 1'] != human_comp['Word 2']]
        human_comp = human_comp.to_records(index=False)

        self.human_top_sim_353 = list(human_comp)

        simlex = pd.read_csv("SimLex-999/SimLex-999_formatted.csv")
        simlex = simlex.drop(
            columns=['POS', 'conc(w1)', 'conc(w2)', 'concQ', 'Assoc(USF)', 'SimAssoc333', 'SD(SimLex)'])

        simlex = simlex[simlex["SimLex999"] >= 5.0]

        # There  were obvious things like word1: tiger word2: tiger which had 10/10 similarity, so I cleaned for them.
        simlex = simlex[simlex['word1'] != simlex['word2']]
        simlex["SimLex999"] /= 10
        simlex = simlex.to_records(index=False)
        self.simlex_top_sim_999 = list(simlex)


        self.vocabulary_emma = list(EmmaModel.wv.vocab.keys())
        self.vocabulary_news = list(NewsModel.wv.vocab.keys())

        self.shared_vocab_set = set([item for item in self.vocabulary_emma if item in self.vocabulary_news])

    def record_significant_matches(self, w1, w2, sim_score, sim_name):
            self.result[(w1, w2)][sim_name] = sim_score

    def update_similarity_dict(self):
        for (w1, w2, sim) in self.human_top_sim_353:
            if w1 in self.shared_vocab_set and w2 in self.shared_vocab_set:
                self.record_significant_matches(w1, w2, sim, "353_sim")
                news_out = NewsModel.model_news.wv.similarity(w1, w2)
                emma_out = EmmaModel.model_emma.wv.similarity(w1, w2)
                self.record_significant_matches(w1, w2, news_out, "news_out")
                self.record_significant_matches(w1, w2, emma_out, "emma_out")

        for (w1, w2, sim) in self.simlex_top_sim_999:
            if w1 in self.shared_vocab_set and w2 in self.shared_vocab_set:
                self.record_significant_matches(w1, w2, sim, "999_sim")
                news_out = NewsModel.model_news.wv.similarity(w1, w2)
                emma_out = EmmaModel.model_emma.wv.similarity(w1, w2)
                self.record_significant_matches(w1, w2, news_out, "news_out")
                self.record_significant_matches(w1, w2, emma_out, "emma_out")

    def sort_result(self):

        self.sim_pairs_353 = [pair for pair in self.result.items() if "353_sim" in pair[1] and "999_sim" not in pair[1]]
        self.sim_pairs_999 = [pair for pair in self.result.items() if "999_sim" in pair[1] and "353_sim" not in pair[1]]
        self.sim_pairs_both = [pair for pair in self.result.items() if "999_sim" in pair[1] and "353_sim" in pair[1]]

        self.sim_pairs_353 = sorted(self.sim_pairs_353, key = lambda  x: -x[1]["353_sim"])
        self.sim_pairs_999 = sorted(self.sim_pairs_999, key=lambda x: -x[1]["999_sim"])

    def plot_heatmap(self, data, sort_col = None, save_fig = None, isdict = False):
        if not isdict:
            data = dict(data)

        data_df = pd.DataFrame.from_dict(data, orient='index')
        if sort_col:
            data_df = data_df.sort_values(by=sort_col, ascending = False)
        ax = sns.heatmap(data_df, cmap="YlGnBu")
        fig = ax.get_figure()
        fig.set_size_inches(10, 10)
        fig.show()
        if save_fig:
            fig.savefig(save_fig)




if __name__ == "__main__":
    model = similarityComparison()
    model.update_similarity_dict()
    model.sort_result()
    model.plot_heatmap(model.sim_pairs_353, "353_sim", "figs/353_sim.png")
    model.plot_heatmap(model.sim_pairs_999, "999_sim", "figs/999_sim.png")
