#!/usr/bin/env python3
"""
Enhanced GO Term Query Script with Propagation and Command-Line Interface

Usage:
    python query_go_terms_enhanced.py --terms GO:0008237,GO:0008236 --fdr 0.05 --aspect MF
    python query_go_terms_enhanced.py --terms GO:0008237 --fdr 0.05,0.10,0.20 --propagate
"""

import pandas as pd
import numpy as np
import glob
import os
import pickle
import json
import argparse
from pathlib import Path

def load_ancestors_data():
    """Load GO ancestors and metadata dictionaries"""
    base_path = "/Users/harsh/Documents/PFP_PUBLISH/raw_data_from_uniprot"
    
    ancestors_data = {}
    json_data = {}
    
    for aspect in ['function', 'process', 'component']:
        ancestors_file = f"{base_path}/{aspect}_ALL_data_ancestors_dict.json"
        with open(ancestors_file) as f:
            ancestors_data[aspect] = json.load(f)
        
        json_file = f"{base_path}/{aspect}_ALL_data_json_dict.json"
        with open(json_file) as f:
            json_data[aspect] = json.load(f)
    
    return ancestors_data, json_data

def load_gene_dictionaries():
    """Load all gene dictionaries with PFP predictions"""
    PICKLE_DIR = "genomes_to_annotate_with_PlasmoFP/with_PFP_predictions_complete_2"
    gene_dicts = {}
    for pkl in glob.glob(f"{PICKLE_DIR}/*_gene_dict_with_PFP.pkl"):
        key = os.path.basename(pkl).split("_gene_dict")[0]
        with open(pkl, "rb") as f:
            gene_dicts[key] = pickle.load(f)
    return gene_dicts

def get_propagated_terms(target_terms, ancestors_data, aspect):
    """Get all ancestor terms for propagation"""
    propagated_terms = set(target_terms)
    
    for term in target_terms:
        if term in ancestors_data[aspect]:
            ancestors = ancestors_data[aspect][term]
            propagated_terms.update(ancestors)
    
    return propagated_terms

def get_term_name(term_id, json_data, aspect):
    """Get human-readable name for a GO term"""
    try:
        aspect_map = {'MF': 'function', 'BP': 'process', 'CC': 'component'}
        mapped_aspect = aspect_map.get(aspect, aspect.lower())
        
        if term_id in json_data[mapped_aspect]:
            return json_data[mapped_aspect][term_id].get('name', f'Unknown term {term_id}')
        else:
            return f'Unknown term {term_id}'
    except:
        return f'Unknown term {term_id}'

def query_go_terms_enhanced(go_terms, species_list=None, target_fdr=0.05, aspect="MF", 
                           include_fdr_comparison=False, propagate=False, 
                           ancestors_data=None, json_data=None, output_protein_ids=False, 
                           protein_ids_dir=None):
    """    
    go_terms : list
        List of GO IDs to query
    species_list : list, optional
        List of species to analyze. If None, analyzes all species
    target_fdr : float or list, default 0.05
        FDR threshold(s) to use for PlasmoFP predictions
    aspect : str, default "MF"
        Which ontology aspect to analyze (MF, BP, CC)
    include_fdr_comparison : bool, default False
        If True, includes multiple FDR levels in the analysis
    propagate : bool, default False
        If True, includes ancestor terms in the analysis
    ancestors_data : dict, optional
        Ancestors data for propagation
    json_data : dict, optional
        JSON metadata for term names
    output_protein_ids : bool, default False
        If True, outputs protein IDs to text files
    protein_ids_dir : str, optional
        Directory to save protein ID files
    """
    gene_dicts = load_gene_dictionaries()
    
    if propagate and ancestors_data:
        aspect_map = {'MF': 'function', 'BP': 'process', 'CC': 'component'}
        mapped_aspect = aspect_map[aspect]
        propagated_terms = get_propagated_terms(go_terms, ancestors_data, mapped_aspect)
        print(f"Original terms: {len(go_terms)}")
        print(f"After propagation: {len(propagated_terms)}")
        print(f"Added ancestor terms: {len(propagated_terms) - len(go_terms)}")
    else:
        propagated_terms = set(go_terms)
    
    go_terms_dict = {}
    aspect_map = {'MF': 'function', 'BP': 'process', 'CC': 'component'}
    mapped_aspect = aspect_map.get(aspect, aspect.lower())
    
    for term in go_terms:
        if json_data:
            go_terms_dict[term] = get_term_name(term, json_data, aspect)
        else:
            go_terms_dict[term] = f"Term {term}"
    
    if species_list is None:
        species_list = list(gene_dicts.keys())
    
    if include_fdr_comparison or isinstance(target_fdr, list):
        if isinstance(target_fdr, list):
            fdr_levels = target_fdr
        else:
            fdr_levels = [0.01, 0.05, 0.10, 0.20, 0.30]
    else:
        fdr_levels = [target_fdr]
    
    if aspect == "MF":
        curated_slots = ['GO Function', 'GO IEA Function']
        pfp_slot = 'PFP MF'
    elif aspect == "BP":
        curated_slots = ['GO Process', 'GO IEA Process']
        pfp_slot = 'PFP BP'
    elif aspect == "CC":
        curated_slots = ['GO Component', 'GO IEA Component']
        pfp_slot = 'PFP CC'
    else:
        raise ValueError("aspect must be 'MF', 'BP', or 'CC'")
    
    species_to_name = {
        'PlasmoDB-68_Pfalciparum3D7': 'P. falciparum',
        'PlasmoDB-68_PvivaxSal1': 'P. vivax',
        'PlasmoDB-68_PmalariaeUG01': 'P. malariae',
        'PlasmoDB-68_PovalecurtisiGH01': 'P. ovale curtisi',
        'PlasmoDB-68_PovalewallikeriPowCR01': 'P. ovale wallikeri',
        'PlasmoDB-68_PknowlesiH': 'P. knowlesi',
        'PlasmoDB-68_PbergheiANKA': 'P. berghei',
        'PlasmoDB-68_Pchabaudichabaudi': 'P. chabaudi',
        'PlasmoDB-68_Pyoeliiyoelii17XNL2023': 'P. yoelii',
        'PlasmoDB-68_Pgallinaceum8A': 'P. gallinaceum',
        'PlasmoDB-68_PreichenowiCDC': 'P. reichenowi',
        'PlasmoDB-68_PadleriG01': 'P. adleri',
        'PlasmoDB-68_PblacklockiG01': 'P. blacklocki',
        'PlasmoDB-68_PcoatneyiHackeri': 'P. coatneyi',
        'PlasmoDB-68_PcynomolgiM': 'P. cynomolgi',
        'PlasmoDB-68_PfragileNilgiri': 'P. fragile',
        'PlasmoDB-68_PgaboniG01': 'P. gaboni',
        'PlasmoDB-68_PinuiSanAntonio1': 'P. inui',
        'PlasmoDB-68_PvinckeibrucechwattiDA': 'P. vinckei'
    }
    
    rows = []
    
    protein_ids_data = {} if output_protein_ids else None
    if output_protein_ids and protein_ids_dir:
        Path(protein_ids_dir).mkdir(parents=True, exist_ok=True)
    
    for sp in species_list:
        if sp not in gene_dicts:
            print(f"Warning: {sp} not found in gene dictionaries")
            continue
            
        genes = gene_dicts[sp]
        
        for go_id, go_name in go_terms_dict.items():
            curated_count = 0
            curated_genes = set()
            
            for gene_id, ann in genes.items():
                curated_set = set()
                for slot in curated_slots:
                    curated_set.update(ann.get(slot, []))
                
                if propagate:
                    term_intersection = curated_set.intersection(propagated_terms)
                    if go_id in term_intersection or (go_id in propagated_terms and term_intersection):
                        curated_count += 1
                        curated_genes.add(gene_id)
                else:
                    if go_id in curated_set:
                        curated_count += 1
                        curated_genes.add(gene_id)
            
            for fdr in fdr_levels:
                combined_genes = set(curated_genes) 
                pfp_genes = set()  
                
                for gene_id, ann in genes.items():
                    if pfp_slot in ann:
                        pfp_data = ann[pfp_slot]
                        if fdr in pfp_data:
                            predictions = pfp_data[fdr]
                            pfp_terms = set()
                            
                            for go_tuple in predictions:
                                if isinstance(go_tuple, tuple) and len(go_tuple) >= 1:
                                    pfp_terms.add(go_tuple[0])
                                elif isinstance(go_tuple, str):
                                    pfp_terms.add(go_tuple)
                            
                            if propagate:
                                term_intersection = pfp_terms.intersection(propagated_terms)
                                if go_id in term_intersection or (go_id in propagated_terms and term_intersection):
                                    combined_genes.add(gene_id)
                                    if gene_id not in curated_genes:
                                        pfp_genes.add(gene_id)
                            else:
                                if go_id in pfp_terms:
                                    combined_genes.add(gene_id)
                                    if gene_id not in curated_genes:
                                        pfp_genes.add(gene_id)
                
                combined_count = len(combined_genes)
                
                if output_protein_ids:
                    key = f"{species_to_name.get(sp, sp.replace('PlasmoDB-68_', ''))}_{go_id}_{fdr}"
                    protein_ids_data[key] = {
                        'curated_genes': sorted(list(curated_genes)),
                        'pfp_genes': sorted(list(pfp_genes)),
                        'all_genes': sorted(list(combined_genes)),
                        'go_name': go_name,
                        'species': species_to_name.get(sp, sp.replace('PlasmoDB-68_', '')),
                        'propagated': propagate
                    }
                
                if curated_count > 0:
                    pct = (combined_count - curated_count) / curated_count * 100
                else:
                    pct = float('inf') if combined_count > 0 else 0.0
                
                if pct == float('inf'):
                    pct_display = 'New'
                elif pct == 0.0:
                    pct_display = 0.0
                else:
                    pct_display = round(pct, 2)
                
                rows.append({
                    'Species': species_to_name.get(sp, sp.replace('PlasmoDB-68_', '')),
                    'GO_ID': go_id,
                    'GO_Name': go_name,
                    'FDR_Level': fdr,
                    'Curated/IEA': curated_count,
                    'Curated+PFP': combined_count,
                    'PFP_Only': combined_count - curated_count,
                    'PctGrowth': pct_display,
                    'Propagated': propagate
                })
    
    if len(fdr_levels) > 1:
        columns = ['Species', 'GO_ID', 'GO_Name', 'FDR_Level', 'Curated/IEA', 'Curated+PFP', 'PFP_Only', 'PctGrowth', 'Propagated']
    else:
        columns = ['Species', 'GO_ID', 'GO_Name', 'Curated/IEA', 'Curated+PFP', 'PFP_Only', 'PctGrowth', 'Propagated']
        for row in rows:
            del row['FDR_Level']
    
    df = pd.DataFrame(rows, columns=columns)
    
    if output_protein_ids and protein_ids_data and protein_ids_dir:
        for key, data in protein_ids_data.items():
            filename = f"{key.replace(':', '-').replace(' ', '_')}.txt"
            filepath = Path(protein_ids_dir) / filename
            
            with open(filepath, 'w') as f:
                f.write(f"# GO Term: {data['go_name']} ({key.split('_')[-2]})\n")
                f.write(f"# Species: {data['species']}\n")
                f.write(f"# FDR Level: {key.split('_')[-1]}\n")
                f.write(f"# Propagated: {data['propagated']}\n")
                f.write(f"# Generated by query_go_terms_enhanced.py\n")
                f.write(f"#\n")
                f.write(f"# Curated genes: {len(data['curated_genes'])}\n")
                f.write(f"# PFP-only genes: {len(data['pfp_genes'])}\n")
                f.write(f"# Total genes: {len(data['all_genes'])}\n")
                f.write(f"#\n")
                
                if data['curated_genes']:
                    f.write("# === CURATED GENES ===\n")
                    for gene in data['curated_genes']:
                        f.write(f"{gene}\n")
                    f.write("#\n")
                
                if data['pfp_genes']:
                    f.write("# === PFP-ONLY GENES ===\n")
                    for gene in data['pfp_genes']:
                        f.write(f"{gene}\n")
                    f.write("#\n")
                
                f.write("# === ALL GENES (CURATED + PFP) ===\n")
                for gene in data['all_genes']:
                    f.write(f"{gene}\n")
        
        print(f"\nProtein ID files saved to: {protein_ids_dir}")
        print(f"Generated {len(protein_ids_data)} protein ID files")
    
    return df

def main():
    parser = argparse.ArgumentParser(
        description='Query GO terms across Plasmodium species with PFP predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --terms GO:0008237,GO:0008236 --fdr 0.05 --aspect MF
  %(prog)s --terms GO:0008237 --fdr 0.05,0.10,0.20 --aspect MF
  %(prog)s --terms GO:0008237 --fdr 0.05 --aspect MF --propagate
  %(prog)s --terms-file go_terms.txt --fdr 0.05 --aspect MF --human-malaria-only
  %(prog)s --terms GO:0008237 --fdr 0.05 --aspect MF --output-protein-ids --protein-ids-dir output/
  %(prog)s --terms GO:0006412 --fdr 0.05 --aspect BP --species "P. falciparum,P. vivax"
        """
    )
    
    parser.add_argument('--terms', 
                       help='Comma-separated list of GO IDs to query (e.g., GO:0008237,GO:0008236)')
    
    parser.add_argument('--terms-file', 
                       help='Text file containing GO terms (one GO ID per line, e.g., GO:0008237)')
    
    parser.add_argument('--fdr', default='0.05',
                       help='FDR threshold(s) to use. Single value or comma-separated (default: 0.05)')
    
    parser.add_argument('--aspect', choices=['MF', 'BP', 'CC'], default='MF',
                       help='GO aspect: MF (Molecular Function), BP (Biological Process), CC (Cellular Component)')
    
    parser.add_argument('--species', default=None,
                       help='Comma-separated list of species to analyze (default: all species)')
    
    parser.add_argument('--propagate', action='store_true',
                       help='Include ancestor terms in the analysis (propagation)')
    
    parser.add_argument('--output', default=None,
                       help='Output CSV file (default: print to stdout)')
    
    parser.add_argument('--human-malaria-only', action='store_true',
                       help='Analyze only human malaria parasites')
    
    parser.add_argument('--output-protein-ids', action='store_true',
                       help='Output protein IDs to text files')
    
    parser.add_argument('--protein-ids-dir', default='protein_ids_output',
                       help='Directory to save protein ID files (default: protein_ids_output)')
    
    args = parser.parse_args()
    
    if not args.terms and not args.terms_file:
        parser.error("Either --terms or --terms-file must be provided")
    
    if args.terms and args.terms_file:
        parser.error("Cannot use both --terms and --terms-file. Choose one.")
    
    if args.terms:
        go_terms = [term.strip() for term in args.terms.split(',')]
    else:
        try:
            with open(args.terms_file, 'r') as f:
                lines = f.readlines()
            
            go_terms = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                if '\t' in line:
                    go_id = line.split('\t')[0].strip()
                elif ',' in line:
                    go_id = line.split(',')[0].strip()
                else:
                    go_id = line.strip()
                
                if go_id.startswith('GO:') and len(go_id) >= 10:
                    go_terms.append(go_id)
                else:
                    print(f"Warning: Skipping invalid GO ID format: '{go_id}'")
            
            go_terms = list(dict.fromkeys(go_terms))
            
            if not go_terms:
                parser.error(f"No valid GO terms found in {args.terms_file}")
            
            print(f"Loaded {len(go_terms)} GO terms from {args.terms_file}")
        except Exception as e:
            parser.error(f"Error reading terms file: {e}")
    
    fdr_str = args.fdr.split(',')
    if len(fdr_str) == 1:
        target_fdr = float(fdr_str[0])
        include_fdr_comparison = False
    else:
        target_fdr = [float(f) for f in fdr_str]
        include_fdr_comparison = True
    
    if args.human_malaria_only:
        species_list = [
            'PlasmoDB-68_Pfalciparum3D7',
            'PlasmoDB-68_PvivaxSal1',
            'PlasmoDB-68_PmalariaeUG01',
            'PlasmoDB-68_PovalecurtisiGH01',
            'PlasmoDB-68_PovalewallikeriPowCR01'
        ]
    elif args.species:
        name_to_species = {
            'P. falciparum': 'PlasmoDB-68_Pfalciparum3D7',
            'P. vivax': 'PlasmoDB-68_PvivaxSal1',
            'P. malariae': 'PlasmoDB-68_PmalariaeUG01',
            'P. ovale curtisi': 'PlasmoDB-68_PovalecurtisiGH01',
            'P. ovale wallikeri': 'PlasmoDB-68_PovalewallikeriPowCR01'
        }
        species_names = [name.strip() for name in args.species.split(',')]
        species_list = []
        for name in species_names:
            if name in name_to_species:
                species_list.append(name_to_species[name])
            elif name.startswith('PlasmoDB-68_'):
                species_list.append(name)
            else:
                print(f"Warning: Unknown species '{name}'")
    else:
        species_list = None
    
    ancestors_data = None
    json_data = None
    if args.propagate:
        print("Loading GO ancestors data...")
        try:
            ancestors_data, json_data = load_ancestors_data()
            print("✓ Ancestors data loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load ancestors data: {e}")
            print("Proceeding without propagation...")
            args.propagate = False
    
    print(f"Analyzing {len(go_terms)} GO terms: {', '.join(go_terms)}")
    print(f"FDR level(s): {args.fdr}")
    print(f"Aspect: {args.aspect}")
    print(f"Propagation: {'Enabled' if args.propagate else 'Disabled'}")
    
    df = query_go_terms_enhanced(
        go_terms=go_terms,
        species_list=species_list,
        target_fdr=target_fdr,
        aspect=args.aspect,
        include_fdr_comparison=include_fdr_comparison,
        propagate=args.propagate,
        ancestors_data=ancestors_data,
        json_data=json_data,
        output_protein_ids=args.output_protein_ids,
        protein_ids_dir=args.protein_ids_dir if args.output_protein_ids else None
    )
    
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")
    else:
        print("\n" + "="*100)
        print(df.to_string(index=False))
    
    print(f"\n=== SUMMARY ===")
    print(f"Total rows: {len(df)}")
    if 'FDR_Level' in df.columns:
        print(f"FDR levels analyzed: {sorted(df['FDR_Level'].unique())}")
    print(f"Species analyzed: {len(df['Species'].unique())}")
    print(f"GO terms analyzed: {len(df['GO_ID'].unique())}")

if __name__ == "__main__":
    main()
