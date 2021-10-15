from rocket_qa.model.src.rocket_rerank import RocketReRanker

def rerank(self,docs: DocumentArray, **kwargs):
    for d in docs:
        data = np.char.add(np.char.add(d.text+'\t', d.matches.get_attributes('text')),'\t0')
        new_scores = self.model.get_scores(data)
        for match,new_score in zip(d.matches,new_scores):
            match.score = new_score
        d.matches.sort(key=lambda x: x.score.value,reverse=True)