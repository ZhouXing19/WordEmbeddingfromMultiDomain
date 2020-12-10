import NewsModel
import EmmaModel
import PersuasionModel
import MobyModel
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


        self.EmmaModel = EmmaModel
        self.NewsModel = NewsModel
        self.PersuasionModel = PersuasionModel
        self.MobyModel = MobyModel

    def record_significant_matches(self, w1, w2, sim_score, sim_name):
            self.result[(w1, w2)][sim_name] = sim_score

    def update_similarity_dict(self, model1, model2, human_model, model1_name, model2_name, human_model_name):

        vocab1 = list(model1.wv.vocab.keys())
        vocab2 = list(model2.wv.vocab.keys())
        shared_vocab_set = set([item for item in vocab1 if item in vocab2])
        print( "Share : " + str(2 * len(shared_vocab_set) / (len(vocab1) + len(vocab2))))


        for (w1, w2, sim) in human_model:
            if w1 in shared_vocab_set and w2 in shared_vocab_set:
                self.record_significant_matches(w1, w2, sim, human_model_name)
                m1_out = model1.wv.similarity(w1, w2)
                m2_out = model2.wv.similarity(w1, w2)
                self.record_significant_matches(w1, w2, m1_out, model1_name)
                self.record_significant_matches(w1, w2, m2_out, model2_name)


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

    EmmaModel = model.EmmaModel
    NewsModel = model.NewsModel
    MobyModel = model.MobyModel
    PersuasionModel = model.PersuasionModel

    Human353 = model.human_top_sim_353
    Human999 = model.simlex_top_sim_999

    model.update_similarity_dict(EmmaModel, NewsModel, Human353, "Emma", "News", "353_sim")
    model.update_similarity_dict(EmmaModel, NewsModel, Human999, "Emma", "News", "999_sim")

    model.sort_result()
    model.plot_heatmap(model.sim_pairs_353, "353_sim", "figs/353_sim_emma_news.png")
    model.plot_heatmap(model.sim_pairs_999, "999_sim", "figs/999_sim_emma_news.png")


    model = similarityComparison()
    model.update_similarity_dict(EmmaModel, MobyModel, Human353, "Emma", "Moby", "353_sim")
    model.update_similarity_dict(EmmaModel, MobyModel, Human999, "Emma", "Moby", "999_sim")
    model.sort_result()
    model.plot_heatmap(model.sim_pairs_353, "353_sim", "figs/353_sim_emma_moby.png")
    model.plot_heatmap(model.sim_pairs_999, "999_sim", "figs/999_sim_emma_moby.png")


    model = similarityComparison()
    model.update_similarity_dict(EmmaModel, PersuasionModel, Human353, "Emma", "Persuasion", "353_sim")
    model.update_similarity_dict(EmmaModel, PersuasionModel, Human999, "Emma", "Persuasion", "999_sim")
    model.sort_result()
    model.plot_heatmap(model.sim_pairs_353, "353_sim", "figs/353_sim_emma_persuasion.png")
    model.plot_heatmap(model.sim_pairs_999, "999_sim", "figs/999_sim_emma_persuasion.png")

    model = similarityComparison()
    model.update_similarity_dict(MobyModel, NewsModel, Human353, "Moby", "News", "353_sim")
    model.update_similarity_dict(MobyModel, NewsModel, Human999, "Moby", "News", "999_sim")
    model.sort_result()
    model.plot_heatmap(model.sim_pairs_353, "353_sim", "figs/353_sim_moby_news.png")
    model.plot_heatmap(model.sim_pairs_999, "999_sim", "figs/999_sim_moby_news.png")

    model = similarityComparison()
    model.update_similarity_dict(PersuasionModel, NewsModel, Human353, "Persuasion", "News", "353_sim")
    model.update_similarity_dict(PersuasionModel, NewsModel, Human999, "Persuasion", "News", "999_sim")
    model.sort_result()
    model.plot_heatmap(model.sim_pairs_353, "353_sim", "figs/353_sim_persuasion_news.png")
    model.plot_heatmap(model.sim_pairs_999, "999_sim", "figs/999_sim_persuasion_news.png")




