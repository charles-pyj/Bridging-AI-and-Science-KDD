import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from adjustText import adjust_text
from scipy.stats import linregress
df = pd.read_csv("../table_statistics/science_kdd_type.csv")
with open("../results/instruction_embedding/cluster_labels/name_kdd_sci.json","r") as f:
    GPT_summary = json.load(f)
# df = pd.read_csv("../table_statistics/AI_rank_by_sci_ratio.csv")
# with open("../results/instruction_embedding/cluster_labels/top_words_ai_GPT_fixed.json","r") as f:
#     GPT_summary = json.load(f)
x = df['size_total'].values
y = df['AI_size_total'].values
# y = df['sci_size_total'].values
# AI_from_sci = df['sci_from_sci'].values
# AI_from_AI = df['sci_from_AI'].values
ind = df['cluster_idx'].values
# y = df['sci_size_total'].values
plt.figure(figsize=(15, 10))
# from scipy.stats import linregress
sns.regplot(x=x, y=y, data=df, fit_reg=False, scatter_kws={'color': '#FFC300','s':75})
# Step 2: Calculate the regression line and confidence intervals
slope, intercept, r_value, p_value, std_err = linregress(x, y)
ylim = 1.05*0.52
xlim = 1.05
x_vals = np.linspace(min(x), max(x)*xlim, 100)
y_vals = intercept + slope * x_vals
# Calculate the confidence interval
t_val = 1.96  # For a 95% confidence interval
ci = t_val * std_err
print(f"Stats: ",slope, intercept,std_err)
y_vals_upper = y_vals + ci * x_vals
y_vals_lower = y_vals - ci * x_vals
lower_indices = [ind[i] for i in range(len(y)) if y[i] < intercept + (slope-ci) * x[i]]
higher_indices = [ind[i] for i in range(len(y)) if y[i] > intercept + (slope+ci) * x[i]]
print(f"Confidence: {ci}")
print(f"Slope: {slope}")
# Step 3: Plot the regression line and confidence intervals
plt.plot(x_vals, y_vals, color='black', label=f'Regression Line: y = {intercept:.2f} + {slope:.2f}x')
plt.fill_between(x_vals, y_vals_lower, y_vals_upper, color='grey', alpha=0.2, label='95% Confidence Interval')
# Assuming x and y are your variables, and df is your DataFrame
slope, intercept, r_value, p_value, std_err = linregress(x, y)

# print(f"Intercept: {intercept}")
# print(f"Slope: {slope}")
# slope, _, _, _ = np.linalg.lstsq(x[:, np.newaxis], y, rcond=None)
# sns.scatterplot(x=x, y=y)
# plt.plot(x, slope * x, color='red')  # regression line

# Plot the custom line with dashed line
texts = []
print(len(x))
print(len(ind))
plt.xlabel('# of publications in each cluster',fontsize=20)
# plt.ylabel('# of publications addressing scientific challenges')
plt.ylabel('# of AI4Science publications in each cluster',fontsize=20)
for i in range(len(x)):
    ratio = 3
    if ((x[i] > 200 and y[i] > 50 and y[i] < max(y) * ylim) or (x[i] > 670 and y[i] < 50 and y[i] < max(y) * ylim)): 
        # and "Magnetic" not in GPT_summary[ind[i]] and "Evolution" not in GPT_summary[ind[i]] and "Dynamical" not in GPT_summary[ind[i]] and "Imaging" not in GPT_summary[ind[i]] # Customize these values as needed
        text = df['GPT_summarization'][i]
        label_color = 'black'  # You can add your conditional color logic here
        texts.append(plt.text(x[i]+3, y[i], text, fontsize=5, ha='left', color=label_color))
        # if AI_from_sci[i] / AI_from_AI[i] > 3:
        #     plt.text(x[i]+12, y[i], GPT_summary[ind[i]], fontsize=6, ha='left',color='black')
        # elif AI_from_AI[i] / AI_from_sci[i] > 3:
        #     plt.text(x[i]+12, y[i], GPT_summary[ind[i]], fontsize=6, ha='left',color='black')
        # else:
        #     plt.text(x[i]+12, y[i], GPT_summary[ind[i]], fontsize=6, ha='left',color='black')
# set y lim
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylim([min(y), max(y)* ylim  ])
plt.xlim([min(x) , max(x) * xlim ])
indices = [i for i in range(len(df)) if (df['size_total'][i] > max(x) * xlim) or (df['AI_size_total'][i] > max(y)*ylim)]

print(len(indices))
for i in indices:
    print(df.iloc[i]['GPT_summarization'])
    print(df.iloc[i]['size_total'],df.iloc[i]['AI_size_total'])
adjust_text(texts,  # Slightly adjust the expansion for texts
            force_points=0.005,        # Forces the texts closer to the points
            force_text=0.005, )
# in log scale
# plt.xscale('log')
# plt.yscale('log')
# for i in range(len(y)):
#      if y[i] > max(y) * ylim:
#           print(GPT_summary[ind[i]])
#           print((x[i],y[i]))
#plt.savefig("science_fitted_line_kdd_labeled.pdf",dpi=1000,format="pdf",bbox_inches='tight')
# plt.show()
# print(len(lower_indices))
# for i in higher_indices:
#     print(GPT_summary[i])
# print("="*100)
# for i in lower_indices:
#     print(GPT_summary[i])

# with open("../results/instruction_embedding/cluster_labels/under_sci_kdd.json","w") as f:
#     json.dump([int(i) for i in lower_indices] ,f)
# with open("../results/instruction_embedding/cluster_labels/over_sci_kdd.json","w") as f:
#     json.dump([int(i) for i in higher_indices] ,f)

# df = pd.read_csv("../table_statistics/AI_rank_by_sci_ratio.csv")
# with open("../results/instruction_embedding/cluster_labels/top_words_ai_GPT_fixed.json","r") as f:
#     GPT_summary = json.load(f)
# print(GPT_summary[279])
# # df = pd.read_csv("../table_statistics/AI_rank_by_sci_ratio.csv")
# # with open("../results/instruction_embedding/cluster_labels/top_words_ai_GPT_fixed.json","r") as f:
# #     GPT_summary = json.load(f)
# x = df['size_total'].values
# # y = df['AI_size_total'].values
# # AI_from_sci = df['AI_from_sci'].values
# # AI_from_AI = df['AI_from_AI'].values
# y = df['sci_size_total'].values
# AI_from_sci = df['sci_from_sci'].values
# AI_from_AI = df['sci_from_AI'].values
# ind = df['cluster_idx'].values
# summary = df['GPT_summarization'].values
# indices = [i for i in range(len(summary)) if summary[i] not in ["Deep Learning","Machine Learning"]]
# x_new = [x[i] for i in indices]
# y_new = [y[i] for i in indices]
# summary = [summary[i] for i in indices]
# # y = df['sci_size_total'].values

# plt.figure(figsize=(10, 7))

# #sns.scatterplot(x=x_new, y=y_new, label='Data Points')
# sns.regplot(x=x_new, y=y_new, data=df, fit_reg=True)
# # Step 2: Calculate the regression line and confidence intervals
# slope, intercept, r_value, p_value, std_err = linregress(x_new, y_new)
# x_vals = np.linspace(min(x_new), max(x_new), 100)
# y_vals = intercept + slope * x_vals

# # Calculate the confidence interval
# t_val = 1.96  # For a 95% confidence interval
# ci = t_val * std_err
# y_vals_upper = y_vals + ci * x_vals
# y_vals_lower = y_vals - ci * x_vals
# lower_indices = [ind[i] for i in indices if y[i] < intercept + (slope-ci) * x[i]]

# print(f"Confidence: {ci}")
# print(f"Slope: {slope}")
# # Step 3: Plot the regression line and confidence intervals
# plt.plot(x_vals, y_vals, color='blue', label=f'Regression Line: y = {intercept:.2f} + {slope:.2f}x')
# plt.fill_between(x_vals, y_vals_lower, y_vals_upper, color='blue', alpha=0.2, label='95% Confidence Interval')

# # Step 4: Customize labels and plot texts
# plt.xlabel('# of publications in each cluster')
# plt.ylabel('# of publications solving scientific problems')
# for i in range(len(x)):
#     if GPT_summary[ind[i]] not in ["Deep Learning", "Machine Learning"]:
#         if (x[i] > 100 and y[i] > 100) or (x[i] > 300 and y[i] < 100):
#             if AI_from_sci[i] / AI_from_AI[i] > 3:
#                 plt.text(x[i] + 12, y[i], GPT_summary[ind[i]], fontsize=6, ha='left', color='green')
#             elif AI_from_AI[i] / AI_from_sci[i] > 3:
#                 plt.text(x[i] + 12, y[i], GPT_summary[ind[i]], fontsize=6, ha='left', color='red')
#             else:
#                 plt.text(x[i] + 12, y[i], GPT_summary[ind[i]], fontsize=6, ha='left', color='brown')

# # Step 5: Save and show the plot
# plt.legend()
# plt.savefig("./AI_fitted_line_with_confidence_interval.jpg", dpi=1000)
# #plt.show()
# with open("../results/instruction_embedding/cluster_labels/under_ai_indices.json","w") as f:
#     json.dump([int(i) for i in lower_indices] ,f)
# print(len(lower_indices))
# for i in lower_indices:
#     print(GPT_summary[i])