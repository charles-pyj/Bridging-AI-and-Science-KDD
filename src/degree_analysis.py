import json
import pandas as pd
import numpy as np

def get_degree(prefix,indices,degree_dict):
    degrees = []
    for i in indices:
        if f"{prefix}_{i}" in degree_dict:
            degrees.append(degree_dict[f"{prefix}_{i}"])

    print(f"Mean degree: {np.mean(degrees)} of {len(degrees)} clusters")

# sci_df = pd.read_csv("../table_statistics/science_kdd_by_degree.csv")
# degree = sci_df['degree']
# ai_df = pd.read_csv("../table_statistics/AI_kdd_degree.csv")
# AI_degree = ai_df['degree']
scientific_bike = pd.read_csv("../results/instruction_embedding/bike_vis/scientific_problem_kdd_hdbscan.tsv",sep = "\t")
ai_bike = pd.read_csv("../results/instruction_embedding/bike_vis/AI_method_kdd_compare.tsv",sep = "\t")
with open("../results/instruction_embedding/cluster_labels/scientific_problems_kdd.json","r") as f:
    cluster_labels_sci = json.load(f)
with open("../results/instruction_embedding/cluster_labels/AI_method_kdd_merged.json","r") as f:
    cluster_labels_ai = json.load(f)
with open("../results/instruction_embedding/cluster_labels/under_sci_kdd.json","r") as f:
    under_sci_indices = json.load(f)
with open("../results/instruction_embedding/cluster_labels/over_sci_kdd.json","r") as f:
    over_sci_indices = json.load(f)
with open("../results/instruction_embedding/cluster_labels/under_ai_kdd.json","r") as f:
    under_ai_indices = json.load(f)
with open("../results/instruction_embedding/cluster_labels/over_ai_kdd.json","r") as f:
    over_ai_indices = json.load(f)
def power_law_fit(
        degrees=None,
        title='',
        xlabel='x',
        ylabel='y',
        savepath = None
    ):
    import powerlaw
    import matplotlib.pyplot as plt

    # Example degree distribution (replace this with your actual data)
    # For this example, I'm using a sample list. Replace it with your real data.

    # Step 1: Fit a power law model
    fit = powerlaw.Fit(degrees, xmin=1)    
    # Step 2: Print out some basic statistics about the fit
    print(f"Alpha (exponent): {fit.alpha}")
    print(f"Xmin (min value for fit): {fit.xmin}")


    print(f"KS Statistic: {fit.power_law.KS()}")

    # Step 4: Likelihood ratio test between power law and exponential distribution
    R, p_value = fit.distribution_compare('power_law', 'exponential')
    print(f"Log-Likelihood Ratio R: {R}")
    print(f"p-value of Log-Likelihood Ratio Test: {p_value}")

    # Step 5: Likelihood ratio test between power law and log-normal distribution
    R_log, p_log = fit.distribution_compare('power_law', 'lognormal')
    print(f"Log-Likelihood Ratio R (Log-Normal): {R_log}")
    print(f"p-value of Log-Likelihood Ratio Test (Log-Normal): {p_log}")

    # Step 3: Create a scatter plot for empirical distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    alpha = fit.alpha
    xmin = fit.xmin

    # Step 3: Calculate the constant C
    # Using the formula: C = (alpha - 1) / xmin^(alpha - 1)
    C = (alpha - 1) / (xmin ** (alpha - 1))
    custom_label = f'C={C:.2f}, alpha={alpha:.2f}, xmin={xmin}'
    # Calculate empirical PDF (Probability Density Function) using numpy
    unique_degrees, counts = np.unique(degrees, return_counts=True)
    pdf = counts / sum(counts)
    
    # Scatter plot of empirical degree distribution
    ax.scatter(unique_degrees, pdf, color='b', label=custom_label, s=20, alpha=0.6)

    # Step 4: Plotting the degree distribution with power law fit
    # Plot the empirical distribution (PDF) using the `powerlaw` library
    #fit.plot_pdf(color='b', linewidth=2, label=None, ax=ax)

    # Plot the power law fit
    fit.power_law.plot_pdf(color='r', linestyle='--', ax=ax, label=f'Power Law fit\nalpha={fit.alpha:.2f}')

    # Add title and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    
    # Save the figure if a path is provided
    if savepath:
        plt.savefig(savepath)
    #plt.show()
    #plt.show()
    # data = list(degrees_dict_ai.values())
    # hist, bin_edges = np.histogram(data, bins=np.logspace(np.log10(min(data)), np.log10(max(data)), 20), density=True)
    # x = (bin_edges[:-1] + bin_edges[1:]) / 2  # Bin centers
    # y = hist

    # # Fit the power law to the histogram data
    # popt, _ = curve_fit(power_law, x, y, maxfev=10000)

    # # Plot the data and the power law fit
    # plt.scatter(x, y, label='Data')
    # plt.plot(x, power_law(x, *popt), label=f'Fit: y={popt[0]:.2f}x^{popt[1]:.2f}', color='r')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel('Value')
    # plt.ylabel('Probability Density')
    # plt.title('Power Law Fit')
    # plt.legend()
    # plt.show()


def plot_degrees_log(degrees, title, savepath, kde_color='red'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Convert degrees dictionary to a list of values
    data = degrees

    # Check for zero or negative values before log transformation
    # if any(data <= 0):
    #     raise ValueError("All degree values must be positive for log transformation.")

    # Log-transform the data
    log_data = np.log10(data)  # Using base-10 logarithm for a comparable plot

    # Create the figure
    plt.figure(figsize=(10, 6))
    log_data_df = pd.DataFrame(log_data, columns=["Log Degree"])
    # Create a histogram and plot KDE using Seaborn
    sns.histplot(log_data, bins=30, color='#5A9', edgecolor='black',kde=False, stat="density")
    sns.kdeplot(log_data, color='crimson',linestyle="--")
    # Customize the title and labels
    plt.title(title, fontsize=20, fontweight='bold')
    plt.xlabel('Log(Degree)', fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, linestyle='-', linewidth=0.3, color='gray')
    # Save the plot
    plt.savefig(savepath,format="pdf",bbox_inches='tight',dpi=500)
    #plt.show()

def qq_plot_and_ks_test(
    data, 
    title='',
    marker_size=2,  # Specify marker size for Q-Q plot
    savepath = None
):
    # Log-transform the data
    import numpy as np
    import scipy.stats as stats
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(rc={'figure.figsize':(3,3)})
    log_data = np.log(data)
    
    # Generate a Q-Q plot
    _, ax = plt.subplots()
    stats.probplot(
        log_data, 
        dist="norm", 
        plot=ax
    )
    ax.set_title(title)
    ax.set_xlabel('Theoretical Quantiles',fontsize=15)
    ax.set_ylabel('Ordered novelty scores (log)',fontsize=15)
    
    # Adjust marker size after probplot()
    ax.get_lines()[0].set_markersize(marker_size)
    # Perform Kolmogorov-Smirnov test against log-normal distribution
    kstest_result = stats.kstest(data, 'lognorm', stats.lognorm.fit(data))#, args=(np.median(log_data), np.median(abs(log_data - np.median(log_data)))))
    #print(kstest_result)
    shape, loc, scale = stats.lognorm.fit(data)

    # ax.text(0.05, 0.95, f"KS test p-value: {kstest_result.pvalue:.2f} \n shape: {shape:.2f} \n loc: {loc:.2f} \n scale: {scale:.2f}", 
    #         transform=ax.transAxes, fontsize=10, verticalalignment='top',
    #         bbox=dict(facecolor='white', alpha=0.6, edgecolor='black'))
    ax.text(0.05, 0.95, f"KS test p-value: {kstest_result.pvalue:.2f}", 
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.6, edgecolor='black'))
    # ax.text(-2, 6, f'KS test p-value: {kstest_result.pvalue:.4f}', bbox=dict(facecolor='white', alpha=0.5))
    # Show the plot
    plt.savefig(savepath,format="pdf",bbox_inches='tight')
    #plt.show()

# qq_plot_and_ks_test(degrees_dict_sci, title='Q-Q Plot and KS Test',savepath="../visualizations/QQ_ks_sci.pdf")
# qq_plot_and_ks_test(degrees_dict_ai, title='Q-Q Plot and KS Test',savepath="../visualizations/QQ_ks_ai.pdf")

def edges_to_degree(df,indices = None):
    def parse_name(name):
        return int(name.split("_")[-1])
    degree_dict = {}
    start = df['start'].tolist()
    end = df['end'].tolist()
    for i in np.unique(start):
        if indices == None:
            degree_dict[i] = len(df[df['start'] == i])
        else:
            if parse_name(i) in indices:
                degree_dict[i] = len(df[df['start'] == i])
    return degree_dict

sci_df = pd.read_csv("../results/instruction_embedding/edges/kdd_ai_sci_test.csv")
degrees = edges_to_degree(sci_df,under_ai_indices)
print(sum(list(degrees.values()))/len(under_ai_indices))


# sci_df = pd.read_csv("../results/instruction_embedding/edges/kdd_sci_ai_test.csv")
# degrees = edges_to_degree(sci_df,indices=under_sci_indices)
# print(sum(list(degrees.values()))/len(under_sci_indices))
# assert len(scientific_bike) == len(cluster_labels_sci)
# assert len(ai_bike) == len(cluster_labels_ai)
# ai4Sci_indices = [i for i in range(len(scientific_bike)) if scientific_bike.iloc[i]['size'] == 8 and scientific_bike.iloc[i]['year'] >= 2023]
# print(len(ai4Sci_indices))
# print(f"In well areas: {len([i for i in ai4Sci_indices if cluster_labels_sci[i] in over_sci_indices])}")
# print(f"In under areas: {len([i for i in ai4Sci_indices if cluster_labels_sci[i] in under_sci_indices])}")
# ai4Sci_indices = [i for i in range(len(ai_bike)) if ai_bike.iloc[i]['size'] == 8 and ai_bike.iloc[i]['year'] >= 2023]
# print(len(ai4Sci_indices))
# print(f"In well areas: {len([i for i in ai4Sci_indices if cluster_labels_ai[i] in over_ai_indices])}")
# print(f"In under areas: {len([i for i in ai4Sci_indices if cluster_labels_ai[i] in under_ai_indices])}")
sci_df = pd.read_csv("../table_statistics/science_kdd_type.csv")
degree = sci_df['degree']
ai_df = pd.read_csv("../table_statistics/AI_kdd_degree_type.csv")
AI_degree = ai_df['degree']
degree = [d for d in degree if d > 0]
AI_degree = [d for d in AI_degree if d > 0]
# print(len(degree))
# plot_degrees_log(degree,"","../visualizations/log_degree_sci_kde_kdd.pdf")
# plot_degrees_log(AI_degree,"","../visualizations/log_degree_ai_kde_kdd.pdf")
# power_law_fit(title="Science cluster degree power law fit",degrees=degree,savepath="../visualizations/degree_sci_power_law_kdd.jpg")

qq_plot_and_ks_test(degree, title='Q-Q Plot and KS Test',savepath="../visualizations/QQ_ks_sci_kdd.pdf")
qq_plot_and_ks_test(AI_degree, title='Q-Q Plot and KS Test',savepath="../visualizations/QQ_ks_ai_kdd.pdf")