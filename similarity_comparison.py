import NewsModel
import EmmaModel
import io
import collections
import pandas as pd

"""
Method: Take human scores >= 5.0 similar words from human_comp dataset, call it human_top_sim. 
Compare the similarity score of those two words from news model and literature model
to the top 50 most similar words from human_top_sim.
"""

"""
Analysis using WordSimilarity-353
"""
human_comp = pd.read_csv("wordsim353/combined.csv")
human_comp = human_comp[human_comp["Human (mean)"] >= 5.0]
# There  were obvious things like word1: tiger word2: tiger which had 10/10 similarity, so I cleaned for them.
human_comp = human_comp[human_comp['Word 1'] != human_comp['Word 2']]
human_comp = human_comp.to_records(index=False)
human_top_sim = list(human_comp)

vocabulary_emma = list(EmmaModel.wv.vocab.keys())
vocabulary_news = list(NewsModel.wv.vocab.keys())

vocabulary = [item for item in vocabulary_emma if item in vocabulary_news]

shared_vocab_set = set(vocabulary)

def record_significant_matches(w1, w2, sim):
	lit_match = []
	news_match = []

	# human_word: human_sim, news_sim, lit_sim
	sim_reference = collections.defaultdict(list)

	if w1 in shared_vocab_set and w2 in shared_vocab_set:
		news_out = NewsModel.model_news.wv.similarity(w1, w2)
		emma_out = EmmaModel.model_emma.wv.similarity(w1, w2)
		sim_reference[w1].append(w2)
		sim_reference[w1].append(sim)
		sim_reference[w1].append(news_out)
		sim_reference[w1].append(emma_out)
	return sim_reference

complete_353_sim_map = collections.defaultdict(list)

for (w1, w2, sim) in human_top_sim:
	out_353 = record_significant_matches(w1,w2,sim)
	complete_353_sim_map.update(out_353)
print("\n\n Dictionary of 353 Test similaritiy references: ", complete_353_sim_map)

"""
Analysis using SimLex-999
"""
simlex = pd.read_csv("SimLex-999/SimLex-999_formatted.csv")
simlex = simlex.drop(columns=['POS', 'conc(w1)', 'conc(w2)', 'concQ', 'Assoc(USF)', 'SimAssoc333', 'SD(SimLex)'])

simlex = simlex[simlex["SimLex999"] >= 5.0]

# There  were obvious things like word1: tiger word2: tiger which had 10/10 similarity, so I cleaned for them.
simlex = simlex[simlex['word1'] != simlex['word2']]
simlex = simlex.to_records(index=False)
simlex_top_sim = list(simlex)

simlex_complete_sim_map = collections.defaultdict(list)

for (w1, w2, sim) in simlex_top_sim:
	simlex_out = record_significant_matches(w1,w2,sim)
	simlex_complete_sim_map.update(simlex_out)
print("\n\n Dictionary of SimLex-999 Test similaritiy references: ", simlex_complete_sim_map)





















