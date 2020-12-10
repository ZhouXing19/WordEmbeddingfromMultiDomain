import NewsModel
import EmmaModel
import MobyModel
import PersuasionModel
import io
import collections
import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

class top_similar_words:
	def __init__(self, model1, model2):
		self.res = []
		self.sim1 = {}
		self.sim2 = {}
		self.model1 = model1
		self.model2 = model2
		self.table = {}

	def compare_top_similar(self,word):
		out_m1 = set()
		out_m2 = set()

		preprocessed_mod1_similarities = self.model1.wv.most_similar(positive=word, topn=10)
		preprocessed_mod2_similarities = self.model2.wv.most_similar(positive=word, topn=10)

		for i in preprocessed_mod1_similarities:
			out_m1.add(i[0])

		for j in preprocessed_mod2_similarities:
			out_m2.add(j[0])

		shared_words_count = 0
		shared_words = []
		for k in out_m1:
			if k in out_m2:
				shared_words.append(k)
				shared_words_count += 1

		self.sim1[word] = out_m1
		self.sim2[word] = out_m2
		# print ("\n\n News Analysis: Words Most Similar to ", word, ":", out_news)
		# print ("\n\n Novel Analysis: Words Most Similar to ", word, ":", out_novel)
		# print("Shared Similar Words: ", shared_words)
		# print("Shared Similar Words Count: ", shared_words_count)
		return [shared_words_count, shared_words]

	def compare_two_models(self):
		self.vocab1 = list(self.model1.wv.vocab.keys())
		self.vocab2 = list(self.model2.wv.vocab.keys())

		vocabulary = [item for item in self.vocab1 if item in self.vocab2]
		for word in vocabulary:
			shared_words_count, shared_words = self.compare_top_similar(word)
			sim_1 = self.sim1[word]
			sim_2 = self.sim2[word]
			self.res.append((word, shared_words_count, shared_words, sim_1, sim_2))
		self.res = sorted(self.res, key = lambda i: -i[1])

	def show_res(self, model1_name, model2_name, num = None):
		if num == None:
			num = len(self.res)
		assert 0 <= num <= len(self.res)
		for word, shared_words_count, shared_words, sim_1, sim_2 in self.res[:num]:
			print("Current word: ", word)
			print("Shared Words: {}, {} ".format(shared_words_count, shared_words))
			print("Similar words in " + model1_name + " : ", sim_1)
			print("Similar words in " + model2_name + " : ", sim_2)
			print("\n")
	def turn_res_table(self):
		for word, shared_words_count, shared_words, sim_1, sim_2 in self.res:
			self.table[word] = {
				"shared_words_count": shared_words_count,
				"shared_words": shared_words,
				"sim_1": sim_1,
				"sim_2": sim_2
			}

	def get_mean_concreteness(self, word_list, concreteness_dict):
		return np.nanmean([concreteness_dict.get(word, np.nan) for word in word_list])

	def compare_concreteness(self, concreteness_dict, word=None):
		if word is not None:
			print('Concreteness of ' + word + ': '+ str(concreteness_dict.get(word, np.nan)))
			cr_mn_1 = self.get_mean_concreteness(self.table[word]['sim_1'], concreteness_dict)
			print('Concreteness of Top Similar Words in Model 1: '+ str(cr_mn_1))
			cr_mn_2 = self.get_mean_concreteness(self.table[word]['sim_2'], concreteness_dict)
			print('Concreteness of Top Similar Words in Model 2: '+ str(cr_mn_2))
			return None
		else:
			cr_1 = []
			cr_2 = []
			for k,v in self.table.items():
				cr_1.append(self.get_mean_concreteness(v['sim_1'],concreteness_dict))
				cr_2.append(self.get_mean_concreteness(v['sim_2'],concreteness_dict))
			print(stats.ttest_rel(cr_1, cr_2, nan_policy='omit'))
			return (cr_1, cr_2)



if __name__ == "__main__":
	cr_df = pd.read_excel("Concreteness_ratings_Brysbaert_et_al_BRM.xlsx")
	cr_dict = cr_df[['Word','Conc.M']].set_index('Word').to_dict()['Conc.M']

	EmmaNews = top_similar_words(EmmaModel, NewsModel)
	EmmaNews.compare_two_models()
	EmmaNews.turn_res_table()
	EmmaNews.show_res("Emma", "News", 5)
	emma_voc = EmmaNews.vocab1
	news_voc = EmmaNews.vocab2
	print(2*len(EmmaNews.res) / (len(emma_voc) + len(news_voc)))

	EmmaMoby = top_similar_words(EmmaModel, MobyModel)
	EmmaMoby.compare_two_models()
	EmmaMoby.turn_res_table()
	EmmaMoby.show_res("Emma", "Moby", 5)
	emma_voc = EmmaMoby.vocab1
	moby_voc = EmmaMoby.vocab2
	print(2*len(EmmaMoby.res) / (len(emma_voc) + len(moby_voc)))

	EmmaPersuasion = top_similar_words(EmmaModel, PersuasionModel)
	EmmaPersuasion.compare_two_models()
	EmmaPersuasion.turn_res_table()
	EmmaPersuasion.show_res("Emma", "Moby", 5)
	emma_voc = EmmaPersuasion.vocab1
	pers_voc = EmmaPersuasion.vocab2
	print(2*len(EmmaPersuasion.res) / (len(emma_voc) + len(pers_voc)))



	cr_1, cr_2 = EmmaNews.compare_concreteness(cr_dict)

	plt.figure(figsize=(8,6))
	plt.hist(cr_1, bins=100, alpha=0.5, label="Emma")
	plt.hist(cr_2, bins=100, alpha=0.5, label="News")
	plt.xlabel("Data", size=14)
	plt.ylabel("Count", size=14)
	plt.title("Mean Similar-Word-Concreteness Values for Two Models")
	plt.legend(loc='upper right')
	plt.savefig('emmanews_cr.png')

	cr_1, cr_2 = EmmaMoby.compare_concreteness(cr_dict)

	plt.figure(figsize=(8,6))
	plt.hist(cr_1, bins=100, alpha=0.5, label="Emma")
	plt.hist(cr_2, bins=100, alpha=0.5, label="Moby")
	plt.xlabel("Data", size=14)
	plt.ylabel("Count", size=14)
	plt.title("Mean Similar-Word-Concreteness Values for Two Models")
	plt.legend(loc='upper right')
	plt.savefig('emmamoby_cr.png')

	cr_1, cr_2 = EmmaPersuasion.compare_concreteness(cr_dict)

	plt.figure(figsize=(8,6))
	plt.hist(cr_1, bins=100, alpha=0.5, label="Emma")
	plt.hist(cr_2, bins=100, alpha=0.5, label="Persuasion")
	plt.xlabel("Data", size=14)
	plt.ylabel("Count", size=14)
	plt.title("Mean Similar-Word-Concreteness Values for Two Models")
	plt.legend(loc='upper right')
	plt.savefig('emmapersuasion_cr.png')

	# print(EmmaNews.table["love"])
	# print(EmmaMoby.table["love"])
	# print(EmmaPersuasion.table["love"])

	# print(EmmaNews.table["wealth"])
	# print(EmmaMoby.table["wealth"])
	# print(EmmaPersuasion.table["wealth"])

	# print(EmmaNews.table["women"])
	# print(EmmaMoby.table["women"])
	# print(EmmaPersuasion.table["women"])











