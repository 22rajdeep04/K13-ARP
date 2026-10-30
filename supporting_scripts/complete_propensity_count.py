import pandas as pd

#Take input from user for the mutation to be processed
inp = input("Enter the mutation (e.g., R539T): ")
print("Your input mutation is:", inp)

#String Splicing to tokenise
og = inp[0]
mut = inp[-1]
resno = int(inp[1:-1])
print(f"input mutation changes {og} to {mut} at position {resno}")

#Original Sequence from AA no. 1 to 726
kelchseq = "MEGEKVKTKANSISNFSMTYDRESGGNSNSDDKSGSSSENDSNSFMNLTSDKNEKTENNSFLLNNSSYGNVKDSLLESIDMSVLDSNFDSKKDFLPSNLSRTFNNMSKDNIGNKYLNKLLNKKKDTITNENNNINHNNNNNNLTANNITNNLINNNMNSPSIMNTNKKENFLDAANLINDDSGLNNLKKFSTVNNVNDTYEKKIIETELSDASDFENMVGDLRITFINWLKKTQMNFIREKDKLFKDKKELEMERVRLYKELENRKNIEEQKLHDERKKLDIDISNGYKQIKKEKEEHRKRFDEERLRFLQEIDKIKLVLYLEKEKYYQEYKNFENDKKKIVDANIATETMIDINVGGAIFETSRHTLTQQKDSFIEKLLSGRHHVTRDKQGRIFLDRDSELFRIILNFLRNPLTIPIPKDLSESEALLKEAEFYGIKFLPFPLVFCIGGFDGVEYLNSMELLDISQQCWRMCTPMSTKKAYFGSAVLNNFLYVFGGNNYDYKALFETEVYDRLRDVWYVSSNLNIPRRNNCGVTSNGRIYCIGGYDGSSIIPNVEAYDHRMKAWVEVAPLNTPRSSAMCVAFDNKIYVIGGTNGERLNSIEVYEEKMNKWEQFPYALLEARSSGAAFNYLNQIYVVGGIDNEHNILDSVEQYQPFNKRWQFLNGVPEKKMNFGAATLSDSYIITGGENGEVLNSCHFFSPDTNEWQLGPSLLVPRFGHSVLIANI"

#Mutant Builder
print(f"Original amino acid in sequence at position number {resno} is {kelchseq[resno-1]}")
if kelchseq[resno-1] == og:
    mutant = kelchseq[:resno-1] + mut + kelchseq[resno:]
    print(mutant)
    if mutant[resno-1] == mut:
        print(True)
    else:
        print(False)
else:
    print("wrong input")
    raise ValueError("The original amino acid does not match the sequence at the specified position.")

# Chou-Fasman Propensities for Alpha, Beta, and Turn
propensity_table = {
    'A': {'alpha': 1.45, 'beta': 0.97, 'turn': 0.66},
    'C': {'alpha': 0.77, 'beta': 1.30, 'turn': 1.19},
    'D': {'alpha': 0.98, 'beta': 0.80, 'turn': 1.46},
    'E': {'alpha': 1.53, 'beta': 0.26, 'turn': 0.74},
    'F': {'alpha': 1.12, 'beta': 1.28, 'turn': 0.60},
    'G': {'alpha': 0.53, 'beta': 0.81, 'turn': 1.56},
    'H': {'alpha': 1.24, 'beta': 0.71, 'turn': 0.95},
    'I': {'alpha': 1.00, 'beta': 1.60, 'turn': 0.47},
    'K': {'alpha': 1.07, 'beta': 0.74, 'turn': 1.01},
    'L': {'alpha': 1.34, 'beta': 1.22, 'turn': 0.59},
    'M': {'alpha': 1.20, 'beta': 1.67, 'turn': 0.60},
    'N': {'alpha': 0.73, 'beta': 0.65, 'turn': 1.56},
    'P': {'alpha': 0.59, 'beta': 0.62, 'turn': 1.52},
    'Q': {'alpha': 1.17, 'beta': 1.10, 'turn': 0.98},
    'R': {'alpha': 0.79, 'beta': 0.90, 'turn': 0.95},
    'S': {'alpha': 0.79, 'beta': 0.72, 'turn': 1.43},
    'T': {'alpha': 0.82, 'beta': 1.20, 'turn': 0.96},
    'V': {'alpha': 1.14, 'beta': 1.65, 'turn': 0.50},
    'W': {'alpha': 1.14, 'beta': 1.19, 'turn': 0.96},
    'Y': {'alpha': 0.61, 'beta': 1.29, 'turn': 1.14}
}

window_size = 5
def calculate_propensity_dataframe(sequence, window_size):
    
    records = []

    for i in range(len(sequence) - window_size + 1):
        segment = sequence[i:i + window_size]
        alpha_vals = []
        beta_vals = []
        turn_vals = []

        for aa in segment:
            props = propensity_table.get(aa, {'alpha': 0.0, 'beta': 0.0, 'turn': 0.0})
            alpha_vals.append(props['alpha'])
            beta_vals.append(props['beta'])
            turn_vals.append(props['turn'])

        avg_alpha = sum(alpha_vals) / window_size
        avg_beta = sum(beta_vals) / window_size
        avg_turn = sum(turn_vals) / window_size

        records.append({
            'Position': i,
            'Segment': segment,
            'Avg_Alpha': avg_alpha,
            'Avg_Beta': avg_beta,
            'Avg_Turn': avg_turn
        })

    # Load into a DataFrame
    df = pd.DataFrame(records)
    return df
frame_num = resno - window_size

#compute propensity of the mutant strain
df_propensity = calculate_propensity_dataframe(mutant, window_size)

df_select_propensity = df_propensity.iloc[frame_num: frame_num + 5].reset_index(drop=True)
print("\nMutant propensities (changed windows):")
print(df_select_propensity)

#compute propensity of the wild-type 
df_wt = calculate_propensity_dataframe(kelchseq, window_size)

df_wt_select = df_wt.iloc[frame_num: frame_num + 5].reset_index(drop=True)
print("\nWT propensities (same windows):")
print(df_wt_select)

# Mean deviation in the selected frames
cols = ['Avg_Alpha', 'Avg_Beta', 'Avg_Turn']
deviation = (df_select_propensity[cols] - df_wt_select[cols]).abs()

print("\nPer-window absolute deviation:")
print(deviation)

mean_deviation = deviation.mean()
print("\nMean deviation across the 5 changed windows:")
print(mean_deviation.to_string())

#joining dataframes
df_select_merge = pd.merge(df_select_propensity, df_wt_select, on='Position', suffixes=('_mutant', '_wt'), how='outer')
print(df_select_merge)
df_select_merge.to_csv(f'propensity_comparison_{inp}.csv', index=False)