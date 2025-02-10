import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from scipy.stats import linregress
from adjustText import adjust_text

xlim = 1.01 * 0.8
ylim = 1.05 * 0.32

df = pd.read_csv("../table_statistics/AI_kdd_type.csv")
with open("../results/instruction_embedding/cluster_labels/name_kdd_ai.json","r") as f:
    GPT_summary = json.load(f)
print(GPT_summary[279])
x = df['size_total'].values
y = df['sci_size_total'].values
ind = df['cluster_idx'].values
summary = df['GPT_summarization'].values
indices = [i for i in range(len(summary)) if summary[i] not in ["default","default"]]
assert len(indices) == len(summary)
x_new = [x[i] for i in indices]
y_new = [y[i] for i in indices]
summary = [summary[i] for i in indices]
# y = df['sci_size_total'].values

plt.figure(figsize=(15, 10))

#sns.scatterplot(x=x_new, y=y_new, label='Data Points')
sns.regplot(x=x, y=y, data=df, fit_reg=False, scatter_kws={'color': '#8E44AD','s':75})
# Step 2: Calculate the regression line and confidence intervals
slope, intercept, r_value, p_value, std_err = linregress(x, y)
x_vals = np.linspace(min(y), max(x)*xlim, 100)
y_vals = intercept + slope * x_vals

# Calculate the confidence interval
t_val = 1.96  # For a 95% confidence interval
ci = t_val * std_err
y_vals_upper = y_vals + ci * x_vals
y_vals_lower = y_vals - ci * x_vals
lower_indices = [ind[i] for i in indices if y[i] < intercept + (slope-ci) * x[i]]
higher_indices = [ind[i] for i in indices if y[i] > intercept + (slope+ci) * x[i]]
print(f"Confidence: {ci}")
print(f"Slope: {slope}")
print(f"Std: {std_err}")
print(f"intercept: {intercept}")
# Step 3: Plot the regression line and confidence intervals
plt.plot(x_vals, y_vals, color='black', label=f'Regression Line: y = {intercept:.2f} + {slope:.2f}x')
plt.fill_between(x_vals, y_vals_lower, y_vals_upper, color='grey', alpha=0.2, label='95% Confidence Interval')
texts = []
# Step 4: Customize labels and plot texts
plt.xlabel('# of publications in each cluster',fontsize=20)
plt.ylabel('# of AI4Science publications in each cluster',fontsize=20)
for i in range(len(x)):
        if ((x[i] > 65 and y[i] > 50 and y[i] < max(y)*ylim) or (x[i] > 230 and y[i] < 50 and y[i] < max(y)*ylim) or x[i]>500) and "Computational" not in GPT_summary[ind[i]] and "Deep Learning Models" not in GPT_summary[ind[i]] and "Genomic" not in GPT_summary[ind[i]] and "Machine Learning" != GPT_summary[ind[i]]:
            text = df['GPT_summarization'][i]
            label_color = 'black'  # You can add your conditional color logic here
            texts.append(plt.text(x[i]+3, y[i], text, fontsize=5, ha='left', color=label_color))
            # if AI_from_sci[i] / AI_from_AI[i] > 3:
            #     plt.text(x[i] + 3, y[i], GPT_summary[ind[i]], fontsize=8, ha='left', color='green')
            # elif AI_from_AI[i] / AI_from_sci[i] > 3:
            #     plt.text(x[i] + 3, y[i], GPT_summary[ind[i]], fontsize=8, ha='left', color='red')
            # else:
            #     plt.text(x[i] + 3, y[i], GPT_summary[ind[i]], fontsize=8, ha='left', color='brown')
plt.ylim([min(y), max(y) * ylim])
plt.xlim([min(x), max(x) * xlim])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
adjust_text(texts,  # Slightly adjust the expansion for texts
            force_points=0.002,        # Forces the texts closer to the points
            force_text=0.002, )
# Step 5: Save and show the plot
indices = [i for i in range(len(df)) if (df['size_total'][i] > max(x) * xlim) or (df['sci_size_total'][i] > max(y)*ylim)]

print(len(indices))
for i in indices:
    print(df.iloc[i]['GPT_summarization'])
    print(df.iloc[i]['size_total'],df.iloc[i]['sci_size_total'])
#plt.savefig("./AI_fitted_line_kdd_labeled.pdf",dpi=1000,format="pdf",bbox_inches='tight')
# for i in range(len(y)):
#      if y[i] > max(y) * ylim:
#           print(GPT_summary[ind[i]])
#           print((x[i],y[i]))
# plt.show()
# for i in higher_indices:
#     print(GPT_summary[i])
# print("="*100)
# for i in lower_indices:
#     print(GPT_summary[i])
# print(len(lower_indices))
# print(len(higher_indices))
# with open("../results/instruction_embedding/cluster_labels/under_ai_kdd.json","w") as f:
#     json.dump([int(i) for i in lower_indices] ,f)
# with open("../results/instruction_embedding/cluster_labels/over_ai_kdd.json","w") as f:
#     json.dump([int(i) for i in higher_indices] ,f)
# print(len(lower_indices))
# for i in higher_indices:
#     print(GPT_summary[i])