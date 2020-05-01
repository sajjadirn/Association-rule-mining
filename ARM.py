import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import fpgrowth
#Question 1
df = pd.read_csv("./specs/gpa_question1.csv")

#Dropping the count column
df = df.drop(columns="count")

#Converting the dataframe to a list as apriori requires a list as input
df2 = df.values.tolist()

#Using the apriori function
te = TransactionEncoder()
te_ary = te.fit(df2).transform(df2)
df2 = pd.DataFrame(te_ary, columns=te.columns_)
#Performing Apriori with min support as 15%
df3 = apriori(df2, min_support=0.15, use_colnames=True)
df3.to_csv("./output/question1_out_apriori.csv", index=False)
print(df3)
#Generating the association rule with minimum confidence of 0.9
df4 = association_rules(df3, metric = "confidence", min_threshold = 0.9 )
df4.to_csv("./output/question1_out_rules9.csv", index=False)

#Generating the association rule with minimum confidence of 0.7
df5 = association_rules(df3, metric = "confidence", min_threshold = 0.7)
df5.to_csv("./output/question1_out_rules7.csv", index=False)

#Question 2
df6 = pd.read_csv("./specs/bank_data_question2.csv")

#Removing column id
df6 = df6.drop(columns="id")

#Binning and replacing the original columns with the binned values
df6['age'] = pd.cut(df6['age'],3)
df6['income'] = pd.cut(df6['income'],3)
df6['children'] = pd.cut(df6['children'],3)

#Using dummies
s = pd.get_dummies(df6)

#Performing fpgrowth with min support as 20%
fp = fpgrowth(s, min_support=0.2, use_colnames = True)
fp.to_csv("./output/question2_out_fpgrowth.csv", index=False)

#Testing with different confidence values and selecting one
rules = association_rules(fp, metric = "confidence", min_threshold = 0.786)
rules.to_csv("./output/question2_out_rules.csv", index=False)