from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import sys
import csv

##functions for processing proteins
def gravy(protein):
    gravy_values = {
        'A':  1.8 , 
        'R': -4.5,  
        'N': -3.5,  
        'D': -3.5,  
        'C':  2.5,  
        'Q': -3.5,  
        'E': -3.5,  
        'G': -0.4,  
        'H': -3.2,  
        'I':  4.5,  
        'L':  3.8,  
        'K': -3.9,  
        'M':  1.9,  
        'F':  2.8,  
        'P': -1.6,  
        'S': -0.8,  
        'T': -0.7,  
        'W': -0.9,  
        'Y': -1.3,  
        'V':  4.2  
        }

    sequence_gravy = []
    for aminoacid in protein:
        gravy_value = gravy_values.get(aminoacid)
        sequence_gravy.append(gravy_value)
    return round(sum(sequence_gravy)/len(protein),2)

def aliphatic_index(protein):

    a = 2.9
    b = 3.9

    counts = {
        'A': 0, 
        'V': 0,
        'I': 0, 
        'L': 0, 
    }

    total_amino_acids = len(protein)

    for amino_acid in protein:
        if amino_acid in counts:
            counts[amino_acid] += 1

    mole_percent = {aa: (count / total_amino_acids) * 100 for aa, count in counts.items()}

    aliphatic_index = mole_percent['A'] + a * mole_percent['V'] + b * (mole_percent['I'] + mole_percent['L'])

    return round(aliphatic_index,2)

def instability_index(protein):
    x = ProteinAnalysis(protein)
    return(round(x.instability_index(),2))

def molecular_weight(protein):
    x = ProteinAnalysis(protein)
    return(round(x.molecular_weight(),2))
def aromaticity(protein):
    x = ProteinAnalysis(protein)
    return(round(x.aromaticity(),2))
def isoelectric_point(protein):
    x = ProteinAnalysis(protein)
    return(round(x.isoelectric_point(),2))
def charge_at_pH(protein, ph):
    x = ProteinAnalysis(protein)
    return round(x.charge_at_pH(ph),2)


def process_file(input_file, output_file):
    with open(input_file, 'r') as input_f:
        output = []
        stability = ''
        for record in SeqIO.parse(input_f, "fasta"):
            protein= str(record.seq)

            if (instability_index(protein))<40:
                stability = "Stable"
            else:
                stability = "Unstable"


            output.append({
                'protein':str(protein),
                'length':len(str(protein)),
                'gravy':gravy(protein),
                'aliphatic_index':aliphatic_index(protein),
                'instability_index':instability_index(protein),
                'molecular_weight': molecular_weight(protein),
                'aromaticity':aromaticity(protein),
                'isoelectric_point':isoelectric_point(protein),
                'charge_ph_7':charge_at_pH(protein, 7)

                })        

        if output:
            csv_content = []
            csv_content.append(['Protein',
                                'Length',
                                'GRAVY',
                                'Aliphatic index',
                                'Instability index',
                               'Molecular weight',
                                'Aromaticity',
                                'Isoelectric point',
                                'Charge at pH 7'])
            for sequence in output:
                csv_content.append([sequence['protein'],
                                    len(sequence['protein']),
                                    sequence['gravy'],
                                    sequence['aliphatic_index'],
                                    sequence['instability_index'],
                                    sequence['molecular_weight'],
                                    sequence['aromaticity'],
                                    sequence['isoelectric_point'],
                                    sequence['charge_ph_7']
                                    ])


        with open(output_file, 'w', newline='') as output_f:
            writer = csv.writer(output_f)
            
            for row in csv_content:
                writer.writerow(row)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python mpp.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    process_file(input_file, output_file)

