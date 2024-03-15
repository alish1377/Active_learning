import numpy as np
from scipy.stats import entropy


class Strategies:
    def __init__(self,
                 method='lc'  # least confident
                 ):
        self.method = method

    def get_high_low_indices(self, sorted_indices, uncertain_size, batch_size):
        high_limit = int(np.ceil((batch_size * (1 - uncertain_size))))
        low_limit = batch_size - high_limit
        if uncertain_size != 1:
            high_confidence = sorted_indices[-high_limit:]
        else:
            high_confidence = np.array([])
        low_confidence = sorted_indices[:low_limit]
        result_indices = np.concatenate((high_confidence, low_confidence)).astype(int)
        return result_indices

    def _get_scores(self, model, dataset, **uncertainty_measure_kwargs):
        scores = None
        if self.method == 'lc':
            class_wise_uncertainty = model.predict(dataset, **uncertainty_measure_kwargs)
            # for each point, select the maximum uncertainty
            scores = 1 - np.max(class_wise_uncertainty, axis=1)
        elif self.method == 'en':
            predictions = model.predict(dataset, **uncertainty_measure_kwargs)
            scores = entropy(predictions, axis=1)
        elif self.method == 'ms':
            predictions = model.predict(dataset)
            scores = np.diff(-np.sort(predictions)[:, ::-1][:, :2]).reshape(len(predictions))
        return scores

    def make_query(self, model, dataset, query_batch_size=1, uncertain_size=1, **uncertainty_measure_kwargs):
        result = None
        if self.method == 'lc':  # least confident
            uncertainty = self._get_scores(model, dataset, **uncertainty_measure_kwargs)
            result = np.argsort(-uncertainty)[:query_batch_size]
        elif self.method == 'random':
            number_of_rows = len(dataset.img_ids)
            result = np.random.choice(number_of_rows, size=query_batch_size, replace=False)
        elif self.method == 'en':   # entropy_strategy
            entropy_arr = self._get_scores(model, dataset, **uncertainty_measure_kwargs)
            indices = np.argsort(-entropy_arr)
            result = self.get_high_low_indices(indices, uncertain_size, query_batch_size)
        elif self.method == 'ms':   # margin_sampling
            margin_diff = self._get_scores(model, dataset, **uncertainty_measure_kwargs)
            indices = np.argsort(margin_diff)
            result = self.get_high_low_indices(indices, uncertain_size, query_batch_size)
        return result
