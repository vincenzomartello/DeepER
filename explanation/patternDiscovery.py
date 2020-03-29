import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth,association_rules


def _appendPrefix(attribute,prefix):
    return list(map(lambda s:prefix+s,str(attribute).split()))


def _allStartsWith(s,prefix):
    for el in s:
        if not(el.startswith(prefix)):
            return False
    return True


def _preProcessNeighbors(nn,opposite_label_data,attribute):
    nn_values = pd.merge(nn,opposite_label_data,left_on=attribute,right_on='id')
    left_values, right_values  = nn_values['ltable_'+attribute],nn_values['rtable_'+attribute]
    left_values_prefixed = list(map(lambda att:(_appendPrefix(att,'L_')),left_values))
    right_values_prefixed = list(map(lambda att:(_appendPrefix(att,'R_')),right_values))
    return list(map(lambda l,r:l+r,left_values_prefixed,right_values_prefixed))


def mineRules(nn_df,oppositeLabelData,attribute,min_confidence,min_support):
	transactions = _preProcessNeighbors(nn_df,oppositeLabelData,attribute)
	te = TransactionEncoder()
	te_ary = te.fit(transactions).transform(transactions)
	df = pd.DataFrame(te_ary, columns=te.columns_)
	frequent_itemsets = fpgrowth(df, min_support=min_support,use_colnames=True)
	ar = association_rules(frequent_itemsets, metric="confidence", min_threshold = min_confidence)
	ar['antecedents_isleft'] = ar['antecedents'].apply(lambda s:_allStartsWith(s,'L_'))
	ar['consequents_isright'] = ar['consequents'].apply(lambda s:_allStartsWith(s,'R_'))
	important_rules = ar[(ar.antecedents_isleft==True)& (ar.consequents_isright==True)]
	return important_rules


