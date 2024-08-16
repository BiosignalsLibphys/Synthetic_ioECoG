import pickle
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd 
import seaborn as sns
import math
import torch
#from ECoG_GAN_dim import Generator

## Fidelity and Authenticity Metrics ##
from scipy.stats import kurtosis, skew, pearsonr, wasserstein_distance, entropy
from scipy.spatial.distance import jensenshannon
from scipy.signal import welch, cwt, morlet2
from skimage.metrics import structural_similarity as ssim
from scipy.stats import mode, entropy, kurtosis, skew, iqr, pearsonr
from scipy.integrate import simps

## Diversity Metrics ##
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


# Functions to generate synthetic timeseries
def generate_synthetic_series(generator, n_series, latent_dim, device):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(n_series, latent_dim).to(device)
        synthetic_series = generator(z)
    return synthetic_series.cpu().numpy()


# Loading the Generator model and synthesizing data
def load_model(g, sd, ld, l_signals, n_signals, device='cpu'):
    """
    -----
    Brief
    -----
    Loading the pre-trained generator model .
    ----------
    Parameters
    ----------
    g : Generator Model
        class
    sd : model parameters
        .pth file path

    l_signals : int
        length of the signals to be generated

    ld : int
        latent dimensions

    n_signals : int
        number of signals to generate
    device : string
        device on which the tensor will be allocated.
    Returns
    -------
    synth_data : nd-array
    """
    generator = g(l_signals,ld,100)
    print('generator initialized')
    generator.load_state_dict(torch.load(sd, map_location=torch.device(device)))
    synth_data = generate_synthetic_series(generator, n_signals, ld, device).reshape(n_signals,l_signals)
    return synth_data

############################################
#saved_state_dict = torch.load(trained_parameters,map_location=torch.device('cpu'))

# Print keys of the saved state dictionary
#print("Saved state dict keys:")
#print(saved_state_dict.keys())

# Print keys of the model's state dictionary
#print("Model state dict keys:")
#print(Generator(2048,100,100).state_dict().keys())

def medium_wave(segment):
    """
    -----
    Brief
    -----
    Compute the mean of all timeseries at the same point.
    ----------
    Parameters
    ----------
    segment : nd-array or list
        Timeseries of the same lenght to be averaged.
    Returns
    -------
    mean_list : list
        mean value at each sample.
    std_list : list
        standard deviation at each sample.
    """
    mean_list = []
    std_list = []
    segment = np.array(segment)
    segment_t = segment.transpose()
    for i in range(len(segment_t)):
        mean = np.mean(segment_t[i])
        std = np.std(segment_t[i])
        mean_list.append(mean)
        std_list.append(std)
    return mean_list, std_list


def calculate_num_bins(data):
    """
    -----
    Brief
    -----
    Calculate number of bins using the Freedman-Diaconis rule.
    ----------
    Parameters
    ----------
    data : nd-array
        Input signal.
    Returns
    -------
    num_bins : int
        Number of bins to build histogram.
    """
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    bin_width = 2 * iqr / len(data) ** (1 / 3)
    num_bins = int(np.ceil((np.max(data) - np.min(data)) / bin_width))
    return num_bins


###FIDELITY

#### HISTOGRAM ANALYSIS #### >>> OR TIME ANALYSIS?
def time_analysis(real_data, synthetic_data):
    """
    -----
    Brief
    -----
    Prints the time statistics for the input real data and for the input synthetic data.
    These include:
        - Mean
        -Standard Deviation
        - Maximum
        - Minimum
        - Kurtosis
        - Skewness
        - Correlation
    ----------
    Parameters
    ----------
    real_data : nd-array or list
        Input real signals.
    synthetic_data : nd-array or list
        Input synthetic signals.
    """
    # Multiple Signal Time Analysis
    if isinstance(real_data,(np.ndarray, list)) and np.array(real_data).ndim >= 2:
        real_data, _ = medium_wave(real_data)
        synthetic_data, _ = medium_wave(synthetic_data)
        print('Dataset Analysis')
    else:
        print('Sample Analysis')

    # Signal Time Analysis
    print('Mean_Real:', np.mean(real_data))
    print('Mean_Synthetic:', np.mean(synthetic_data))
    print('STD_Real:', np.std(real_data))
    print('STD_Synthetic:', np.std(synthetic_data))
    print('Max_Real:', np.max(real_data))
    print('Max_Synthetic:', np.max(synthetic_data))
    print('Min_Real:', np.min(real_data))
    print('Min_Synthetic:', np.min(synthetic_data))
    print('Kurtosis_Real:', kurtosis(real_data))
    print('Kurtosis_Synthetic:', kurtosis(synthetic_data))
    print('Skewness_Real:', skew(real_data))
    print('Skewness_Synthetic:', skew(synthetic_data))
    correlation, _ = pearsonr(real_data, synthetic_data)
    print('Correlation:', correlation)


def wasserstein_distance_(time_series1, time_series2, num_bins=30, range_bins=(0, 1)):
    """
    -----
    Brief
    -----
    Computes the Earth Movers Distance between the distribution of two timeseries or the mean distance between
    multiple timeseries.
    ----------
    Parameters
    ----------
    time_series1 : nd-array or list
        Input signals 1.
    time_series2 : nd-array or list
        Input signals 2.
    num_bins : int
        Number of bins of the histogram. Default value is 30.
    range_bins : tuple
        The lower and upper range of the bins. Default is (0,1).

    Returns
    -------
    wasserstein_dist : float
        The mean Wasserstein distance between the input timeseries distributions.
    """
    # Convert to list if input is a single array, or if list only contains one signal, nest it into another list.
    if isinstance(time_series1, (np.ndarray, list)) and np.array(time_series1).ndim == 1:
        time_series1 = [time_series1]
        print('Sample Analysis')
    else:
        print('Dataset Analysis')
    if isinstance(time_series2, (np.ndarray, list)) and np.array(time_series2).ndim == 1:
        time_series2 = [time_series2]

    wasserstein_dist = []
    n_signals1 = len(time_series1)
    n_signals2 = len(time_series2)

    for sig1 in range(n_signals1):
        for sig2 in range(n_signals2):
            hist_1, bin_edges_1 = np.histogram(time_series1[sig1], bins=num_bins, range=range_bins, density=True)
            hist_2, bin_edges_2 = np.histogram(time_series2[sig2], bins=num_bins, range=range_bins, density=True)
            bin_midpoints = (bin_edges_1[:-1] + bin_edges_1[1:]) / 2
            wasserstein_dist_ = wasserstein_distance(bin_midpoints, bin_midpoints, u_weights=hist_1, v_weights=hist_2)
            wasserstein_dist.append(wasserstein_dist_)

    mean_wasserstein_dist = np.mean(wasserstein_dist)
    std_wasserstein_dist = np.std(wasserstein_dist)

    print('WD:', mean_wasserstein_dist, 'STD:', std_wasserstein_dist)

    return mean_wasserstein_dist


def kl_divergence(time_series1, time_series2, num_bins=30, range_bins=(0, 1)):
    """
    -----
    Brief
    -----
    Compute the Kullback-Leibler divergence (difference between two probability distributions) between two time series.

    ----------
    Parameters
    ----------
    time_series1: 1D array-like, first time series.
    time_series2: 1D array-like, second time series.
    num_bins: int
        number of bins for the histograms.
    range_bins: tuple
        Range of the bins.

    Returns
    -------
    kl_divergence: float
        the KL divergence between the two distributions.
    """
    # Check if the input is a single signal or a list of signals
    if isinstance(time_series1[0], (list, np.ndarray)):
        print('Dataset Analysis')
    else:
        time_series1 = [time_series1]
        print('Sample Analysis')

    # Compute histograms
    hist_1, bin_edges_1 = np.histogram(time_series1, bins=num_bins, range=range_bins, density=True)
    hist_2, bin_edges_2 = np.histogram(time_series2, bins=num_bins, range=range_bins, density=True)

    # Add a small value to avoid division by zero or log of zero
    epsilon = 1e-10
    hist_1 += epsilon
    hist_2 += epsilon

    # Normalize histograms to form probability distributions
    hist_1 /= np.sum(hist_1)
    hist_2 /= np.sum(hist_2)

    # Compute KL divergence
    kl_div = entropy(hist_1, hist_2)

    print("Kullback-Leibler Divergence:", kl_div)

    return kl_div


def js_divergence(time_series1, time_series2, num_bins=30, range_bins=(0, 1)):
    """
    -----
    Brief
    -----
    Compute the Jensen-Shannon (JS) Distance (measure of the similarity between two probability distributions)
    between two time series. This is a symmetric and smoothed version of the KL divergence

    ----------
    Parameters
    ----------
    time_series1: 1D array-like
        first time series.
    time_series2: 1D array-like
        second time series.
    num_bins: int
        number of bins for the histograms.
    range_bins: tuple
        range of the bins.

    Returns:
    -------
    js_divergence: float
        the JS distance between the two distributions.
    """
    # Check if the input is a single signal or a list of signals
    if isinstance(time_series1[0], (list, np.ndarray)):
        print('Dataset Analysis')
    else:
        time_series1 = [time_series1]
        print('Sample Analysis')

    # Compute histograms
    hist_1, bin_edges_1 = np.histogram(time_series1, bins=num_bins, range=range_bins, density=True)
    hist_2, bin_edges_2 = np.histogram(time_series2, bins=num_bins, range=range_bins, density=True)

    # Add a small value to avoid division by zero or log of zero
    epsilon = 1e-10
    hist_1 += epsilon
    hist_2 += epsilon

    # Normalize histograms to form probability distributions
    hist_1 /= np.sum(hist_1)
    hist_2 /= np.sum(hist_2)

    # Compute KL divergence
    js_div = jensenshannon(hist_1, hist_2)

    print("Jensen-Shannon Distance:", js_div)
    return js_div


def hellinger_distance(time_series1, time_series2, num_bins = 30, range_bins = (0,1)):
    """
    -----
    Brief
    -----
    The Hellinger Distance ranges from 0 to 1, where 0 indicates perfect similarity between distributions,
    and 1 is maximum dissimilarity.

    ----------
    Parameters
    ----------
    p: 1d-array
        distribution 1.
    q: 1d-array
        distribution 2.
    Returns:
    -------
    sosq / math.sqrt(2): float
        The Hellinger distance between the two distributions.
    """

    # Check if the input is a single signal or a list of signals
    if isinstance(time_series1[0], (list, np.ndarray)):
        print('Dataset Analysis')
    else:
        time_series1 = [time_series1]
        print('Sample Analysis')

    # Compute histograms
    p, bin_edges_1 = np.histogram(time_series1, bins=num_bins, range=range_bins, density=True)
    q, bin_edges_2 = np.histogram(time_series2, bins=num_bins, range=range_bins, density=True)

    # Add a small value to avoid division by zero or log of zero
    epsilon = 1e-10
    p += epsilon
    q += epsilon

    # Normalize histograms to form probability distributions
    p /= np.sum(p)
    q /= np.sum(q)
    list_of_squares = []
    for p_i, q_i in zip(p, q):
        # caluclate the square of the difference of ith distribution elements
        s = (math.sqrt(p_i) - math.sqrt(q_i)) ** 2

        # append
        list_of_squares.append(s)

    # calculate sum of squares
    sosq = sum(list_of_squares)
    print('Helinger Distance :',sosq / math.sqrt(2))
    return sosq / math.sqrt(2)


def bhattacharyya_distance(time_series1, time_series2, num_bins, range_bins):
    """
    -----
    Brief
    -----
    The Bhattacharyya Distance ,measures the overlap between two probability distributions.
    ----------
    Parameters
    ----------
    time_series1 : 1d-array
        distribution 1.
    time_series2 : 1d-array
        distribution 2.
    num_bins : int
        Number of bins for the histogram
    range_bins : tuple
        The lower and upper range of the bins. Default is (0,1).
    Returns
    -------
    -np.log(bht): float
        The Bhattacharyya distance between the two distributions of the timeseries.
    """

    # Check if the input is a single signal or a list of signals
    if isinstance(time_series1[0], (list, np.ndarray)):
        print('Dataset Analysis')
    else:
        print('Sample Analysis')

    # Compute histograms
    hist_1, bin_edges_1 = np.histogram(time_series1, bins=num_bins, range=range_bins, density=True)
    hist_2, bin_edges_2 = np.histogram(time_series2, bins=num_bins, range=range_bins, density=True)

    bht = 0
    cX = np.concatenate((np.array(time_series1), np.array(time_series2)))
    print(cX)
    for i in range(num_bins):
        p1 = hist_1[i]
        p2 = hist_2[i]
        bht += math.sqrt(p1*p2) * (max(cX) - min(cX))/num_bins

    if bht == 0:
        print('Bhattacharyya Distance:', float('Inf'))
        return float('Inf')
    else:
        print('Bhattacharyya Distance:', -np.log(bht))
        return -np.log(bht)


#### FREQUENCY ANALYSIS ####

class FrequencyAnalysis:
    def __init__(self, fs=2048):
        """
        -----
        Brief
        -----
        Initialize the FrequencyAnalysis with sampling frequency.

        ----------
        Parameters
        ----------
        fs : int
            Sampling frequency of the signals.
        """
        self.fs = fs
        self.real_metrics = None
        self.synthetic_metrics = None

    def compute_relative_power(self, data, data_type):
        """
        -----
        Brief
        -----
        Computes the relative power in different frequency bands for the given data.

        ----------
        Parameters
        ----------
        data : list or np.ndarray
            Input signals to analyze.
        data_type : str
            Type of the data ('real' or 'synthetic').

        -------
        Returns
        -------
        tuple
            Computed frequency bands, power spectral densities, total power, and relative power in different bands.
        """
        freqs = []
        psd = []
        total_power = []
        slow_rel_power = []
        delta_rel_power = []
        theta_rel_power = []
        alpha_rel_power = []
        beta_rel_power = []
        dominant_freq = []
        fs = 2048
        win = 4 * self.fs
        bands = [0.5, 2, 4, 8, 13, 30]
        
        # Check if the input is a single signal or a list of signals
        if isinstance(data[0], (list, np.ndarray)):
            n_signals = len(data)
            print('Dataset Analysis')
        else:
            data = [data]
            n_signals = 1
            print('Sample Analysis')

        for sig in range(n_signals):
            # Compute the Power Spectral Density (PSD) using Welch's method
            freqs_, psd_ = welch(data[sig], self.fs, nperseg=win)
            freqs.append(freqs_)
            psd.append(psd_)
            freq_res = freqs[sig][1] - freqs[sig][0]
            
            # Define frequency bands
            idx_total = np.logical_and(freqs[sig] >= 0, freqs[sig] <= 1024)
            idx_slow = np.logical_and(freqs[sig] >= bands[0], freqs[sig] <= bands[1])
            idx_delta = np.logical_and(freqs[sig] >= bands[1], freqs[sig] <= bands[2])
            idx_theta = np.logical_and(freqs[sig] >= bands[2], freqs[sig] <= bands[3])
            idx_alpha = np.logical_and(freqs[sig] >= bands[3], freqs[sig] <= bands[4])
            idx_beta = np.logical_and(freqs[sig] >= bands[4], freqs[sig] <= bands[5])
            
            # Calculate total power and relative power for each band
            total_power_sig = simps(psd[sig][idx_total], dx=freq_res)
            total_power.append(total_power_sig)
            for band, rel_power in zip([idx_slow, idx_delta, idx_theta, idx_alpha, idx_beta],
                                    [slow_rel_power, delta_rel_power, theta_rel_power, alpha_rel_power, beta_rel_power]):
                power = simps(psd[sig][band], dx=freq_res)
                rel_power.append(power / total_power_sig)
            # Determine the dominant frequency
            dominant_freq.append(freqs[sig][np.argmax(psd[sig])])

        # Print the mean and standard deviation of the relative powers
        print(f'{data_type} mean relative slow power: %.3f perc' % np.mean(slow_rel_power))
        print(f'{data_type} STD slow power: %.3f perc' % np.std(slow_rel_power))
        print(f'{data_type} mean relative delta power: %.3f perc' % np.mean(delta_rel_power))
        print(f'{data_type} STD delta power: %.3f perc' % np.std(delta_rel_power))
        print(f'{data_type} mean relative theta power: %.3f perc' % np.mean(theta_rel_power))
        print(f'{data_type} STD theta power: %.3f perc' % np.std(theta_rel_power))
        print(f'{data_type} mean relative alpha power: %.3f perc' % np.mean(alpha_rel_power))
        print(f'{data_type} STD alpha power: %.3f perc' % np.std(alpha_rel_power))
        print(f'{data_type} mean relative beta power: %.3f perc' % np.mean(beta_rel_power))
        print(f'{data_type} STD beta power: %.3f perc' % np.std(beta_rel_power))
        print(f'{data_type} Mean Dominant frequency: %.3f perc' % np.mean(dominant_freq))
        print(f'{data_type} STD Dominant frequency: %.3f perc' % np.std(dominant_freq))

        return freqs, psd, total_power, slow_rel_power, delta_rel_power, theta_rel_power, alpha_rel_power, beta_rel_power, dominant_freq, idx_slow, idx_delta

    def plot_psd(self, real_data, synthetic_data, x_limit1=0, x_limit2=8, y_limit1=0, y_limit2=0.01):  
        """
        -----
        Brief
        -----
        Plots the power spectral density (PSD) for real and synthetic data.

        ----------
        Parameters
        ----------
        real_data : list or np.ndarray
            Real input signals.
        synthetic_data : list or np.ndarray
            Synthetic input signals.
        x_limit1 : float
            Lower limit for x-axis.
        x_limit2 : float
            Upper limit for x-axis.
        y_limit1 : float
            Lower limit for y-axis.
        y_limit2 : float
            Upper limit for y-axis.

        -------
        Returns
        -------
        None
        """

        # Check if the input is a single signal or a list of signals
        is_sample = not isinstance(real_data[0], (list, np.ndarray)) or not isinstance(synthetic_data[0], (list, np.ndarray))

        # Compute relative power for real and synthetic data
        freqs_r, psd_r, _, slow_rel_power_r, delta_rel_power_r, _, _, _, _, idx_slow, idx_delta = self.compute_relative_power(real_data, 'real')
        freqs_s, psd_s, _, slow_rel_power_s, delta_rel_power_s, _, _, _, _, idx_slow, idx_delta = self.compute_relative_power(synthetic_data, 'synthetic')

        # Plot the PSD for real data
        plt.figure(figsize=(10, 8))
        plt.subplot(121)
        plt.text(5, 0.035, f'Slow: {np.mean(slow_rel_power_r):.2%}', fontsize=12)  # Adjust the position (5, 0.035) as needed
        plt.text(5, 0.032, f'Delta: {np.mean(delta_rel_power_r):.2%}', fontsize=12)  # Adjust the position (5, 0.032) as needed
        f_scale = np.mean(freqs_r, axis=0)
        plt.plot(f_scale, np.mean(psd_r, axis=0), lw=2, color='k')
        plt.fill_between(f_scale, np.mean(psd_r, axis=0), where=idx_slow, color='C1', alpha=0.3)
        plt.fill_between(f_scale, np.mean(psd_r, axis=0), where=idx_delta, color='skyblue')
        plt.xlabel('Frequency (Hz)', fontsize=14)
        plt.ylabel('Power spectral density ($\mu V^2$/Hz)', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlim([x_limit1, x_limit2])
        plt.ylim([y_limit1, y_limit2])  # plt.ylim([0, np.max(psd_r) * 1.1])
        #plt.title("Original", fontsize=14)
        title = 'Original'
        if is_sample:
            title += ' - Sample Analysis'
        
        plt.title(title, fontsize=14)
        plt.legend(["Mean Welch's periodogram", 'Slow Delta Band [0.5-2]Hz', 'Fast Delta Band [2-4]Hz'],fontsize=12)

        # Plot the PSD for synthetic data
        plt.subplot(122)
        plt.text(5, 0.035, f'Slow: {np.mean(slow_rel_power_s):.2%}', fontsize=12)  # Adjust the position (5, 0.035) as needed
        plt.text(5, 0.032, f'Delta: {np.mean(delta_rel_power_s):.2%}', fontsize=12)  # Adjust the position (5, 0.032) as needed
        f_scale = np.mean(freqs_s, axis=0)
        plt.plot(f_scale, np.mean(psd_s, axis=0), lw=2, color='k')
        plt.fill_between(f_scale, np.mean(psd_s, axis=0), where=idx_slow, color='C1', alpha=0.3)
        plt.fill_between(f_scale, np.mean(psd_s, axis=0), where=idx_delta, color='skyblue')
        plt.xlabel('Frequency (Hz)', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlim([x_limit1, x_limit2])
        plt.ylim([y_limit1, y_limit2])  
        #plt.title("Synthetic", fontsize=14)
        title = 'Original'
        if is_sample:
            title += ' - Sample Analysis'
        
        plt.title(title, fontsize=14)
        plt.legend(["Mean Welch's periodogram", 'Slow Delta Band [0.5-2]Hz', 'Fast Delta Band [2-4]Hz'],fontsize=12)

        plt.show() 

    def plot_frequency_comparison(self, real_data, synthetic_data):
        """
        -----
        Brief
        -----
        Plots a bar chart comparing the frequencies of two signals.
        ----------
        Parameters
        ----------
        real_data : list or np.ndarray
            Real input signals.
        synthetic_data : list or np.ndarray
            Synthetic input signals.
        -------
        Returns
        -------
        None
        """ 
        labels = ['slow', 'delta','theta', 'alpha', 'beta']
        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        # Check if the input is a single signal or a list of signals
        is_sample = not isinstance(real_data[0], (list, np.ndarray)) or not isinstance(synthetic_data[0], (list, np.ndarray))

        # Compute relative power for real and synthetic data
        freqs_r, _, _, slow_rel_power_r, delta_rel_power_r, theta_rel_power_r, alpha_rel_power_r, beta_rel_power_r, _, _, _ = self.compute_relative_power(real_data, 'real')
        freqs_s, _, _, slow_rel_power_s, delta_rel_power_s, theta_rel_power_s, alpha_rel_power_s, beta_rel_power_s, _, _, _ = self.compute_relative_power(synthetic_data, 'synthetic')

       # Store the metrics for later use in histogram metrics
        self.real_metrics = {
            'slow': slow_rel_power_r,
            'delta': delta_rel_power_r,
            'theta': theta_rel_power_r,
            'alpha': alpha_rel_power_r,
            'beta': beta_rel_power_r
        }

        self.synthetic_metrics = {
            'slow': slow_rel_power_s,
            'delta': delta_rel_power_s,
            'theta': theta_rel_power_s,
            'alpha': alpha_rel_power_s,
            'beta': beta_rel_power_s
        }

        # Calculate mean and standard deviation for each band
        mean_r = [np.mean(slow_rel_power_r), np.mean(delta_rel_power_r), np.mean(theta_rel_power_r), np.mean(alpha_rel_power_r), np.mean(beta_rel_power_r)]
        std_r = [np.std(slow_rel_power_r), np.std(delta_rel_power_r), np.std(theta_rel_power_r), np.std(alpha_rel_power_r), np.std(beta_rel_power_r)]
        mean_s = [np.mean(slow_rel_power_s), np.std(delta_rel_power_s), np.mean(theta_rel_power_s), np.mean(alpha_rel_power_s), np.mean(beta_rel_power_s)]
        std_s = [np.std(slow_rel_power_s), np.std(delta_rel_power_s), np.std(theta_rel_power_s), np.std(alpha_rel_power_s), np.std(beta_rel_power_s)]

        # Plot the bar chart comparing the frequency bands
        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width/2, mean_r, width, yerr=std_r, capsize=10, label='Real', color='c', alpha=0.7)
        rects2 = ax.bar(x + width/2, mean_s, width, yerr=std_s, capsize=10, label='Synthetic', color='black', alpha=0.7)

        # Add labels, title, and legend
        ax.set_ylabel('Percentage (%)',fontsize=14)
        ax.set_xlabel('Frequency Band',fontsize=14)
        #ax.set_title('Frequency distribution comparison between real and synthetic signals',fontsize=14)
        title = 'Frequency distribution comparison between real and synthetic signals'
        if is_sample:
            title += ' - Sample Analysis'
        
        ax.set_title(title, fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(labels,fontsize=12)
        ax.legend(fontsize=12)

        fig.tight_layout()
        plt.show()

    def print_histogram_metrics(self, data, label):
        """
        -----
        Brief
        -----
        Prints statistical metrics for the given data.

        ----------
        Parameters
        ----------
        data : list or np.ndarray
            Input signals to analyze.
        label : str
            Label to identify the data type (e.g., 'real', 'synthetic').

        -------
        Returns
        -------
        None
        """
        # Check if the input is a list of signals or a single signal
        if isinstance(data[0], (list, np.ndarray)):
            data_combined = np.concatenate(data)
            print('Dataset Analysis')
        else:
            data_combined = data
            print('Sample Analysis')

        # Compute statistical metrics
        mean = np.mean(data_combined)
        median = np.median(data_combined)
        mode_result = mode(data_combined, axis=None)
    
       # Check if mode_result is an array and has elements
        try:
            mode_value = mode_result.mode[0]
        except (IndexError, TypeError):
            mode_value = "undefined"
        data_range = np.ptp(data_combined)
        variance = np.var(data_combined)
        std_dev = np.std(data_combined)
        interquartile_range = iqr(data_combined)
        data_skewness = skew(data_combined)
        data_kurtosis = kurtosis(data_combined)

        # Print the metrics
        print(f"Metrics for {label} data:")
        print(f"Mean: {mean}")
        print(f"Median: {median}")
        print(f"Mode: {mode_value}")
        print(f"Range: {data_range}")
        print(f"Variance: {variance}")
        print(f"Standard Deviation: {std_dev}")
        print(f"Interquartile Range: {interquartile_range}")
        print(f"Skewness: {data_skewness}")
        print(f"Kurtosis: {data_kurtosis}")
        print("")


#### TIME-FREQUENCY ANALYSIS ####

class ScalogramAnalyzer:
    def __init__(self, fs=2048, frequencies=np.linspace(1, 30, 30)):
        """
        -----
        Brief
        -----
        Initialize the ScalogramAnalyzer with sampling frequency and frequency range.

        ----------
        Parameters
        ----------
        fs : int
            Sampling frequency of the signals.
        frequencies : np.array
            Array of frequencies for wavelet transformation.
        """
        self.fs = fs
        self.frequencies = frequencies
        self.scalogram_real = None
        self.scalogram_synthetic = None

    def plot_scalogram(self, real_data, synthetic_data, signal_indice):
        """
        -----
        Brief
        -----
        Compute and plot the scalogram for the given signal index using Morlet wavelet.
        ----------
        Parameters
        ----------
        real_data : np.array or list
            Array or list of real signals.
        synthetic_data : np.array or list
            Array or list of synthetic signals.
        sig : int
            Index of the signal to plot.

        -------
        Returns
        -------
        None
        """

        # Ensure the input data is in the correct form
        real_signal = real_data[signal_indice]
        synthetic_signal = synthetic_data[signal_indice]
     
        # Compute the scalograms
        widths = self.fs / self.frequencies  # Convert frequencies to scales for the CWT
        self.scalogram_real = np.abs(cwt(real_signal, morlet2, widths, w=5.0))  # w=5.0 is a typical choice for the Morlet wavelet
        self.scalogram_synthetic = np.abs(cwt(synthetic_signal, morlet2, widths, w=5.0))
        
        # Plot the scalograms side by side
        time_real = np.linspace(0, self.scalogram_real.shape[1] / self.fs, self.scalogram_real.shape[1])
        time_synthetic = np.linspace(0, self.scalogram_synthetic.shape[1] / self.fs, self.scalogram_synthetic.shape[1])
        
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        
        axs[0].imshow(self.scalogram_real, extent=[time_real.min(), time_real.max(), self.frequencies.min(), self.frequencies.max()], aspect='auto', origin='lower', cmap='terrain')
        axs[0].set_title('Original', fontsize=14)
        axs[0].set_xlabel('Time (s)', fontsize=14)
        axs[0].set_ylabel('Frequency (Hz)', fontsize=14)
        
        axs[1].imshow(self.scalogram_synthetic, extent=[time_synthetic.min(), time_synthetic.max(), self.frequencies.min(), self.frequencies.max()], aspect='auto', origin='lower', cmap='terrain')
        axs[1].set_title('Synthetic', fontsize=14)
        axs[1].set_xlabel('Time (s)', fontsize=14)
        axs[1].set_ylabel('Frequency (Hz)', fontsize=14)
        
        for ax in axs:
            cbar = plt.colorbar(ax.images[0], ax=ax, label='Magnitude')
            cbar.set_label('Magnitude', fontsize=14)
            ax.set_xticks(np.arange(int(time_real.min()), int(time_real.max()) + 1, step=2))
        
        #plt.savefig('scalograms.png')
        plt.show()

    def compute_scalogram_similarity_metrics(self):
        """
        -----
        Brief
        -----
        Compute similarity metrics between the real and synthetic scalograms.
        -------
        Returns
        -------
        mse : float
            Mean Squared Error (MSE) between the real and synthetic scalograms.
        correlation : float
            Pearson Correlation coefficient between the real and synthetic scalograms.
        cos_sim : float
            Cosine Similarity between the real and synthetic scalograms.
        s : float
            Structural Similarity Index (SSIM) between the real and synthetic scalograms.
        """
        if self.scalogram_real is None or self.scalogram_synthetic is None:
            raise ValueError("Scalograms have not been computed. Please run plot_scalogram first.")
        
        # Compute Mean Squared Error (MSE)
        mse = np.mean((self.scalogram_real - self.scalogram_synthetic) ** 2)
        
        # Compute Pearson Correlation
        correlation, _ = pearsonr(self.scalogram_real.flatten(), self.scalogram_synthetic.flatten())

        # Compute Cosine Similarity
        cos_sim = np.dot(self.scalogram_real.flatten(), self.scalogram_synthetic.flatten()) / (np.linalg.norm(self.scalogram_real) * np.linalg.norm(self.scalogram_synthetic))
        
        # Compute Structural Similarity Index (SSIM)
        data_range = max(self.scalogram_real.max() - self.scalogram_real.min(), self.scalogram_synthetic.max() - self.scalogram_synthetic.min())
        s = ssim(self.scalogram_real.astype(np.float64), self.scalogram_synthetic.astype(np.float64), data_range=data_range)
        
        # Print the results
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Pearson Correlation: {correlation}")
        print(f"Cosine Similarity: {cos_sim}")
        print(f"Structural Similarity Index (SSIM): {s}")

        return mse, correlation, cos_sim, s



#### NON-LINEAR ANALYSIS ####

###DIVERSITY

def analyze_data_distribution(real_data, synthetic_data):
    """
    -----
    Brief
    -----
    Analyzes the distribution of real and synthetic data using PCA and t-SNE,
    and visualizes the results in a scatter plot.

    ----------
    Parameters
    ----------
    real_data : list of np.ndarray
        List of arrays where each array is a real signal.
    synthetic_data : list of np.ndarray
        List of arrays where each array is a synthetic signal.

    -------
    Returns
    -------
    None
    """
    # Ensure the input data is in the correct form
    real_data = np.array(real_data)
    synthetic_data = np.array(synthetic_data)

    # Ensure both real_data and synthetic_data have the same number of features
    assert real_data.shape[1] == synthetic_data.shape[1], "Real and synthetic data must have the same number of features."

    # PCA analysis
    pca = PCA(n_components=2)
    pca_real = pd.DataFrame(pca.fit_transform(real_data), columns=['1st Component', '2nd Component'])
    pca_synthetic = pd.DataFrame(pca.transform(synthetic_data), columns=['1st Component', '2nd Component'])

    pca_result = pd.concat([pca_real.assign(Data='Real'), pca_synthetic.assign(Data='Synthetic')])

    # Concatenate real and synthetic data for t-SNE
    tsne_data = np.vstack((real_data, synthetic_data))

    # Debugging info
    print('len real_data:', len(real_data))

    # Ensure perplexity is less than the number of samples
    perplexity_value = min(30, len(tsne_data) - 1)

    # t-SNE analysis
    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity_value)
    tsne_transformed = tsne.fit_transform(tsne_data)

    # Create DataFrame for t-SNE results
    tsne_result = pd.DataFrame(tsne_transformed, columns=['X', 'Y'])
    tsne_result['Data'] = ['Real'] * len(real_data) + ['Synthetic'] * len(synthetic_data)

    # Debugging info
    print(f"Length of tsne_data: {len(tsne_data)}")
    print(tsne_result)

    # Custom colors and alpha values
    palette = {'Real': 'c', 'Synthetic': 'black'}
    alpha = 0.7

    # Plotting the results
    fig, axes = plt.subplots(ncols=2, figsize=(14, 5))

    sns.scatterplot(x='1st Component', y='2nd Component', data=pca_result, hue='Data', palette=palette, style='Data', alpha=alpha, ax=axes[0])
    axes[0].set_title('PCA Result', fontsize=14)
    axes[0].set_xlabel('1st Component', fontsize=14)
    axes[0].set_ylabel('2nd Component', fontsize=14)

    sns.scatterplot(x='X', y='Y', data=tsne_result, hue='Data', palette=palette, style='Data', alpha=alpha, ax=axes[1])
    axes[1].set_title('t-SNE Result', fontsize=14)
    axes[1].set_xlabel('X', fontsize=14)
    axes[1].set_ylabel('Y', fontsize=14)

    # Remove tick marks for a cleaner look
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        sns.despine(ax=ax)

    # Adjust legend font size
    for ax in axes:
        legend = ax.legend(prop={'size': 12})
        legend.set_title('Data', prop={'size': 12})

    # Set a super title for the figure
    #fig.suptitle('Assessing Diversity: Qualitative Comparison of Real and Synthetic Data Distributions', fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=.88)

    # Ensure the plot is displayed
    plt.show()


if __name__ == '__main__':
    # Setting the visualization style
    sns.set_style('white')

    with open('Synthetic Data/generated_signals_EcogGAN_test5.pkl', 'rb') as f:
        data = pickle.load(f)



    ## Generating synthetic series to be compared with the og data ##

    # Defining the trained generator parameters
    latent_dim = 100
    sequence_l = 2048
    device = 'cpu'
    trained_parameters = 'Generators/EcogGAN/generator_EcogGAN_test5.pth'

    # Generate and plot
    n_synthetic_series = 15  # Specify the number of synthetic time series to generate
    #synth_data = load_model(Generator, trained_parameters, sequence_l, latent_dim, n_synthetic_series, device)

    ## Organizing data into dictionary ##
    # Create an empty dictionary
    data_dict = {}
    # Split the data into synthetic and real groups
    synthetic_data = data[:10]  # The first 5 segments are synthetic
    real_data = data[10:]  # The last 5 segments are real
    # Assign the groups to the keys in the dictionary
    data_dict['Synthetic'] = synthetic_data
    data_dict['Real'] = real_data

    # Usage example
    analyze_data_distribution(real_data, synthetic_data)

    ###### Distances ######
    ## Usage on multiple signals ##
    time_analysis(real_data, synthetic_data)
    wasserstein_distance_(real_data, synthetic_data)
    wasserstein_distance_(real_data, real_data)
    wasserstein_distance_(synthetic_data, synthetic_data)

    ## Usage on one signal ##
    time_analysis(real_data[0], synthetic_data[0])
    wasserstein_distance_(np.array(real_data[0]), np.array(synthetic_data[0]))
    kl_divergence(real_data[0], synthetic_data[0])
    js_divergence(real_data[0], synthetic_data[0])
    # Usage
    hellinger_distance(real_data[0], synthetic_data[0])
    # Usage
    bhattacharyya_distance(real_data[0], synthetic_data[0], 30, (0, 1))

    # Usage example

    # Initialize the FrequencyAnalysis class
    fa = FrequencyAnalysis(fs=2048)

    # Compute relative power for real and synthetic data
    fa.compute_relative_power(real_data, 'real')
    fa.compute_relative_power(synthetic_data, 'synthetic')

    # Compute relative power for real and synthetic data - one sample
    fa.compute_relative_power(synthetic_data[0], 'synthetic')

    # Plot power spectral density
    fa.plot_psd(real_data, synthetic_data)

    # Plot power spectral density - one sample
    fa.plot_psd(real_data[0], synthetic_data[0])

    # Plot frequency comparison
    fa.plot_frequency_comparison(real_data, synthetic_data)

    # Print histogram metrics for real data - list of signals
    fa.print_histogram_metrics(real_data, 'real')

    # Plot power spectral density - one sample
    fa.plot_frequency_comparison(real_data[0], synthetic_data[0])

    # Print histogram metrics for real data - one sample
    fa.print_histogram_metrics(real_data[0], 'real')

    # Usage example
    # Assuming real_data and synthetic_data are defined and contain the signal data
    analyzer = ScalogramAnalyzer()
    analyzer.plot_scalogram(real_data, synthetic_data, signal_indice=1)
    mse, correlation, cos_sim, s = analyzer.compute_scalogram_similarity_metrics()




###AUTHENTICITY

#Distance measures >> WD, KL, JS, Hellinger, Bhattacharyya
#Frequency measures >> PSD, relative power
#Time-frequency measures >> Scalogram
#Non-linear measures >> MFDFA
#Diversity measures >> PCA, t-SNE (?)

###UTILITY

#Predictive score (classification model) >> real data vs synthetic data vs real + synthetic data
