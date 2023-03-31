import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
import json

with open('NPHR_config.json', 'rb') as b:
    config = json.load(b)

buckets = config['buckets']
buckets.append(np.inf)

class bucket_sampler():
    def __init__(self, df):
        self.df = df

    def _extract_samples(self):
        tmp = self.df.copy()
        tmp['Log_NET_HEATRATE.CA'] = np.log(tmp['NET_HEATRATE.CA'])
        sample_points = []
        num_samples = 50
        for i in range(num_samples):
            for lower_lim, upper_lim in zip(buckets, buckets[1:]):
                filtered = tmp[(tmp['load']>=lower_lim) &  (tmp['load']<upper_lim)]
                sampled = filtered[filtered['Log_NET_HEATRATE.CA']>filtered['Log_NET_HEATRATE.CA'].quantile(0.75)].sample(1)
                sampled['sample_id'] = f'sampled_{i}'
                sample_points.append(sampled)

        return pd.concat(sample_points), tmp

class separate_anomalies(bucket_sampler):

    def calc_normal_op_equation(self):
        def objective(x, a, b, c):
            return np.exp(a/x + b) + c

        sample_points, tmp = self._extract_samples()

        x = sample_points['load']
        y = sample_points['Log_NET_HEATRATE.CA']

        popt, pcov = curve_fit(objective, x, y)

        self._normal_eq = objective
        self._normal_eq_param = popt
        self._sample_points = sample_points
        self._tmp = tmp
        
        self._tmp['normal_log_nphr'] = self._tmp['load'].apply(lambda x: self._normal_eq(x, *self._normal_eq_param))
        self._tmp['distance_to_normal_log_nphr'] = self._tmp.apply(lambda x: x['Log_NET_HEATRATE.CA'] - x['normal_log_nphr'], axis=1)        

    def visualize_distance(self):
        x = self._sample_points['load']

        x_viz = np.arange(min(x), max(x), 1)
        y_viz = self._normal_eq(x_viz, *self._normal_eq_param)

        fig, axes = plt.subplots(1,2, figsize=(10, 5))
        sns.scatterplot(self._tmp, x='load', y='Log_NET_HEATRATE.CA', ax=axes[0])
        sns.lineplot(x=x_viz, y=y_viz, color='red', linestyle='--', ax=axes[0])
        self._tmp['distance_to_normal_log_nphr'].hist(ax=axes[1])
        
    def separate(self):
        kmeans = KMeans(n_clusters=2)
        self._tmp['cluster'] = kmeans.fit_predict(self._tmp[['distance_to_normal_log_nphr']])
        self._tmp['unusual_operating_condition'] = self._tmp['cluster'].apply(lambda x: 1 if x==kmeans.cluster_centers_.argmin() else 0)
        distance_df = self._tmp[['NET_HEATRATE.CA', 'load', 'unit', 'unusual_operating_condition']]

        return distance_df

    def visualize_clusters(self):
        sns.scatterplot(self._tmp, x='load', y='NET_HEATRATE.CA', hue='unusual_operating_condition')
        plt.show()


class target_calculator():
    def __init__(self, net_heatrate):
        self.net_heatrate = net_heatrate

    def curve_equation(self, x, a, b, c):
        return np.exp(a / x + b) + c

    def calc_expected_nphr(self):
        x = self.net_heatrate['load']
        y = self.net_heatrate['NET_HEATRATE.CA']

        popt, pcov = curve_fit(self.curve_equation, x, y)
        self.eq_params = popt

        self.net_heatrate['expected_nphr'] = self.net_heatrate['load'].apply(lambda x: self.curve_equation(x, *popt))
        self.net_heatrate['NPHR delta'] = self.net_heatrate['NET_HEATRATE.CA'] - self.net_heatrate['expected_nphr']

    def visualize(self):
        x = self.net_heatrate['load']
        x_viz = np.arange(min(x), max(x), 1)
        y_viz = self.curve_equation(x_viz, *self.eq_params)
        # fig, axes = plt.subplots(1, 1, figsize=(10, 8))
        sns.scatterplot(self.net_heatrate, x='load', y='NET_HEATRATE.CA')
        sns.lineplot(x=x_viz, y=y_viz, color='red', linestyle='--')


    def results_dataframe(self):
        return self.net_heatrate