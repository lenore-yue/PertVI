"""Dual-Gate Structural GO/Reactome Cell-Cycle Pathway & Mitotic Quarantine Mask Builder (v0.4.2).

Implements the converged mathematical and biological specification from the PertVI v0.4.2
Elevation Audition:
  1. Decomposes the structural cell-cycle execution machinery into K=6 biological GO/Reactome
     sub-programs (P1: G1/S Origin Licensing; P2: S-Phase Replisome, dNTP & Nucleosome Assembly;
     P3: Intra-S / G2/M Fork Protection & Checkpoint; P4: G2/M Centrosome, Spindle & Kinetochore;
     P5: Mitotic SAC, Sister Chromatid Segregation, APC/C & Cytokinesis; P6: Core Mitotic Residual).
  2. Evaluates signed positive phase induction r_pos^ctrl(g) = max(r_S^ctrl(g), r_G2M^ctrl(g))
     and positive-over-negative phase dominance r_pos^ctrl(g) > |min(r_S^ctrl(g), r_G2M^ctrl(g))|
     strictly on untreated control cells (or all cells if no control mask is provided), preventing
     anti-correlated resting-G1/effector state genes (r_S < -0.15) from being misclassified.
  3. Protects pleiotropic stress, chaperone, UPR, apoptosis, glycolysis, and T-cell lineage/immune
     effectors (G_Protect) from being masked out of state_head or ood_head.
"""

import os
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


GO_PATHWAY_NAMES: Tuple[str, ...] = (
    "P1_G1S_Origin_Licensing",
    "P2_S_Replisome_dNTP_Chromatin",
    "P3_IntraS_G2M_Checkpoint_Repair",
    "P4_G2M_Spindle_Kinetochore",
    "P5_SAC_Segregation_APCC_Cytokinesis",
    "P6_Core_Mitotic_Execution_Residual",
)

# Pleiotropic Stress, Chaperone, UPR, Apoptosis, Core Glycolysis & Immune/Lineage Protection Set
PROTECTED_PLEIOTROPIC_GENES: Set[str] = {
    # T-cell lineage, transcription factors, cytokines, receptors & checkpoints
    "STAT1", "STAT2", "STAT3", "STAT4", "STAT5A", "STAT5B", "STAT6",
    "JAK1", "JAK2", "JAK3", "TYK2",
    "GATA3", "TBX21", "EOMES", "RUNX1", "RUNX2", "RUNX3", "TCF7", "LEF1",
    "ID2", "ID3", "MXD4", "FOXO1", "FOXO3", "BATF", "BATF3", "TOX", "TOX2",
    "IRF1", "IRF2", "IRF4", "IRF7", "IRF8", "IRF9",
    "MYC", "JUN", "JUNB", "JUND", "FOS", "FOSB", "FOSL1", "FOSL2",
    "REL", "RELA", "RELB", "NFKB1", "NFKB2", "NFKBIA", "NFKBIB", "NFKBIZ",
    "NR4A1", "NR4A2", "NR4A3",
    "TNF", "IFNG", "IL2", "IL4", "IL5", "IL10", "IL13", "IL17A", "IL21", "CSF2",
    "IL2RA", "IL2RB", "IL2RG", "IL7R", "IL15RA", "IL4R", "IL6R",
    "GZMA", "GZMB", "GZMH", "GZMK", "GZMM", "PRF1", "GNLY", "NKG7",
    "CCL3", "CCL4", "CCL5", "CXCL8", "CXCL10", "CXCR3", "CXCR4", "CCR7", "SELL",
    "TNFRSF4", "TNFRSF9", "TNFRSF18", "TNFRSF1B", "CD27", "CD28", "ICOS", "CD40LG",
    "PDCD1", "CTLA4", "LAG3", "HAVCR2", "TIGIT", "CD38", "CD44", "CD69",
    "CD3D", "CD3E", "CD3G", "CD247", "LCK", "ZAP70", "LAT", "LCP2", "ITK", "PLCG1",
    "ISG15", "ISG20", "MX1", "MX2", "OAS1", "OAS2", "OAS3", "OASL",
    "IFI6", "IFI16", "IFI27", "IFI35", "IFI44", "IFI44L", "IFIT1", "IFIT2", "IFIT3", "IFITM1", "IFITM2", "IFITM3",
    "TMEM173", "STING1", "BST2", "RSAD2", "HERC5", "USP18",
    # p53, DNA damage arrest, BH3-only & apoptosis effectors
    "TP53", "CDKN1A", "CDKN1B", "CDKN1C", "CDKN2A", "CDKN2B", "CDKN2C", "CDKN2D",
    "MDM2", "MDM4", "GADD45A", "GADD45B", "GADD45G",
    "BAX", "BAK1", "BBC3", "PMAIP1", "BID", "BIM", "BCL2L11",
    "BCL2", "BCL2L1", "BCL2A1", "MCL1", "CFLAR", "FAS", "FASLG", "TNFSF10",
    "CASP3", "CASP7", "CASP8", "CASP9", "APAF1", "ZMAT3", "TIGAR", "RRM2B",
    # ER stress, Unfolded Protein Response (UPR), autophagy & oxidative stress sensors
    "DDIT3", "DDIT4", "ATF3", "ATF4", "ATF5", "ATF6", "XBP1", "ERN1", "EIF2AK3",
    "HSPA5", "HSP90B1", "DNAJB1", "DNAJB9", "DNAJA1", "HERPUD1", "HYOU1", "PDIA4", "PDIA6",
    "SQSTM1", "MAP1LC3B", "GABARAP", "GABARAPL1", "SESN1", "SESN2", "SESN3",
    "TXNIP", "SOD1", "SOD2", "CAT", "GPX1", "GPX4", "PRDX1", "PRDX2", "GCLC", "GCLM", "HMOX1", "NQO1",
    # Heat-shock chaperones, metallothioneins, core glycolysis & cytoskeletal/MHC state markers
    "HSPA1A", "HSPA1B", "HSPA6", "HSPA8", "HSP90AA1", "HSP90AB1", "HSPD1", "HSPE1", "HSPB1", "HSPH1",
    "MT1A", "MT1E", "MT1F", "MT1G", "MT1H", "MT1M", "MT1X", "MT2A",
    "HIF1A", "GAPDH", "LDHA", "LDHB", "PKM", "ENO1", "PGK1", "ALDOA", "TPI1", "GPI", "HK1", "HK2", "PFKP", "SLC2A1", "SLC2A3",
    "AKT1", "MTOR", "RPTOR", "RPS6KB1", "EIF4EBP1", "PIM1", "PIM2", "PIM3",
    "GLRX", "CAPG", "LGALS1", "LGALS3", "S100A4", "S100A6", "S100A10", "S100A11",
    "VIM", "ACTB", "ACTG1", "B2M", "HLA-A", "HLA-B", "HLA-C", "HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1",
}

# Built-in fallback gene sets for P1-P5 (used if local Enrichr GMT cache is absent)
BUILTIN_MODULE_GENES: Dict[str, Set[str]] = {
    "P1_G1S_Origin_Licensing": {
        "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7", "MCM8", "MCM10",
        "CDC6", "CDT1", "ORC1", "ORC2", "ORC3", "ORC4", "ORC5", "ORC6",
        "GINS1", "GINS2", "GINS3", "GINS4", "CDC45", "WDHD1", "RECQL4",
        "CCNE1", "CCNE2", "CDK2", "CDC25A", "E2F1", "E2F2", "E2F3", "E2F7", "E2F8", "TFDP1", "TFDP2",
        "CLSPN", "DSCC1", "DTL", "CHTF18", "CHTF8", "DCC1", "RBL1", "RB1", "SKP2", "FBXO5",
    },
    "P2_S_Replisome_dNTP_Chromatin": {
        "PCNA", "RRM1", "RRM2", "TYMS", "TK1", "DTYMK", "DUT", "DHFR", "DCK", "CMPK1", "NME1", "CTPS1",
        "POLA1", "POLA2", "POLD1", "POLD2", "POLD3", "POLE", "POLE2", "POLE3", "PRIM1", "PRIM2",
        "RFC1", "RFC2", "RFC3", "RFC4", "RFC5", "RPA1", "RPA2", "RPA3",
        "FEN1", "LIG1", "DNA2", "RNASEH2A", "RNASEH2B",
        "CHAF1A", "CHAF1B", "ASF1B", "SLBP", "NASP", "HELLS", "UHRF1", "GMNN", "UNG", "DNMT1",
        "HMGB1", "HMGB2", "HMGB3", "HMGN2", "H2AFX", "H2AFZ", "H2AFV", "H1FX",
        "HIST1H1A", "HIST1H1B", "HIST1H1C", "HIST1H1D", "HIST1H1E",
        "HIST1H2AB", "HIST1H2AC", "HIST1H2AE", "HIST1H2AG", "HIST1H2AI", "HIST1H2AJ", "HIST1H2AK", "HIST1H2AL", "HIST1H2AM",
        "HIST1H2BB", "HIST1H2BD", "HIST1H2BE", "HIST1H2BF", "HIST1H2BG", "HIST1H2BH", "HIST1H2BI", "HIST1H2BJ", "HIST1H2BK", "HIST1H2BM", "HIST1H2BN", "HIST1H2BO",
        "HIST1H3A", "HIST1H3B", "HIST1H3C", "HIST1H3D", "HIST1H3E", "HIST1H3F", "HIST1H3G", "HIST1H3H", "HIST1H3I", "HIST1H3J",
        "HIST1H4A", "HIST1H4B", "HIST1H4C", "HIST1H4D", "HIST1H4E", "HIST1H4F", "HIST1H4H", "HIST1H4I", "HIST1H4J", "HIST1H4K", "HIST1H4L",
        "HIST2H2AA3", "HIST2H2AB", "HIST2H2AC", "HIST2H2BE", "HIST2H2BF", "HIST2H3A", "HIST2H3C", "HIST2H4A", "HIST2H4B",
    },
    "P3_IntraS_G2M_Checkpoint_Repair": {
        "BRCA1", "BRCA2", "RAD51", "RAD51AP1", "RAD51C", "RAD54L", "XRCC2", "XRCC3",
        "EXO1", "BLM", "WRN", "RMI1", "RMI2", "TOP3A",
        "FANCA", "FANCB", "FANCC", "FANCD2", "FANCE", "FANCF", "FANCG", "FANCI", "FANCL", "FANCM",
        "BRIP1", "BARD1", "PALB2", "MSH2", "MSH6", "MLH1", "PMS2",
        "ATAD2", "TIPIN", "TIMELESS", "CHEK1", "CHEK2", "WEE1", "PKMYT1", "ATR", "ATRIP", "HUS1", "RAD1", "RAD9A", "RAD17",
        "FOXM1", "MYBL2", "WHSC1", "NSD2", "PARP1", "PARP2", "BARD1", "TONSL", "MMS22L",
    },
    "P4_G2M_Spindle_Kinetochore": {
        "CDK1", "CCNB1", "CCNB2", "CCNA2", "CDC25B", "CDC25C", "CKS1B", "CKS2",
        "AURKA", "AURKB", "PLK1", "PLK4", "NEK2", "TPX2", "BIRC5", "CDCA8", "INCENP", "BORA",
        "CENPA", "CENPE", "CENPF", "CENPH", "CENPI", "CENPK", "CENPL", "CENPM", "CENPN", "CENPO", "CENPP", "CENPQ", "CENPU", "CENPW", "CENPX",
        "NDC80", "NUF2", "SPC24", "SPC25", "ZWINT", "ZWILCH", "ZW10", "KNTC1", "KNL1", "CASC5",
        "MIS18A", "MIS18BP1", "HJURP", "OIP5", "ITGB3BP", "DSN1", "NSL1", "PMF1",
        "KIF11", "KIF14", "KIF15", "KIF18A", "KIF18B", "KIF20A", "KIF20B", "KIF22", "KIF23", "KIF2C", "KIF4A", "KIFC1",
        "NUSAP1", "PRC1", "DLGAP5", "HMMR", "CKAP2", "CKAP2L", "CKAP5", "GTSE1", "ASPM", "TACC3",
        "SKA1", "SKA2", "SKA3", "SMC2", "SMC4", "NCAPD2", "NCAPD3", "NCAPG", "NCAPG2", "NCAPH", "NCAPH2",
        "TOP2A", "KPNA2", "CEP55", "CEP72", "CEP76", "CEP78", "CEP85", "CEP135", "CEP152", "CEP192",
        "SASS6", "STIL", "CNTROB", "HAUS1", "HAUS2", "HAUS3", "HAUS4", "HAUS5", "HAUS6", "HAUS7", "HAUS8",
    },
    "P5_SAC_Segregation_APCC_Cytokinesis": {
        "MAD2L1", "MAD1L1", "BUB1", "BUB1B", "BUB3", "TTK", "TRIP13", "CMTM7",
        "CDC20", "FZR1", "UBE2C", "UBE2S", "UBE2T", "ANAPC1", "ANAPC4", "ANAPC5", "ANAPC7", "ANAPC10", "ANAPC11", "CDC16", "CDC23", "CDC27",
        "PTTG1", "ESPL1", "RAD21", "SMC1A", "SMC3", "STAG1", "STAG2", "PDS5A", "PDS5B", "WAPL", "CDCA5",
        "SGOL1", "SGOL2", "SGO1", "SGO2", "SPAG5", "KNSTRN", "TROAP",
        "ANLN", "RACGAP1", "ECT2", "CIT", "PBK", "MELK", "CDCA2", "CDCA3", "CDCA7", "MKI67",
        "SEPTIN2", "SEPTIN7", "SEPTIN9", "KIF20A", "KIF23", "PRC1", "AURKB", "INCENP", "BIRC5", "CDCA8",
    },
}


def _parse_gmt_safe(path: str) -> Dict[str, Set[str]]:
    if not path or not os.path.exists(path):
        return {}
    terms: Dict[str, Set[str]] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                name = parts[0]
                genes = {g.split(",")[0].strip() for g in parts[2:] if g.strip()}
                terms[name] = genes
    return terms


def _match_terms(
    db: Dict[str, Set[str]],
    include_patterns: List[str],
    exclude_patterns: Optional[List[str]] = None,
) -> Set[str]:
    out: Set[str] = set()
    for k, v in db.items():
        kl = k.lower()
        if any(p in kl for p in include_patterns):
            if exclude_patterns and any(ep in kl for ep in exclude_patterns):
                continue
            out.update(v)
    return out


def load_structural_go_modules(
    gmt_dir: str = "/home/sihan/.cache/gseapy",
) -> Dict[str, Set[str]]:
    """Loads the 5 structural GO/Reactome cell-cycle execution modules (with built-in fallback)."""
    modules: Dict[str, Set[str]] = {
        k: set(v) for k, v in BUILTIN_MODULE_GENES.items()
    }

    go_bp = _parse_gmt_safe(os.path.join(gmt_dir, "Enrichr.GO_Biological_Process_2026.gmt"))
    go_cc = _parse_gmt_safe(os.path.join(gmt_dir, "Enrichr.GO_Cellular_Component_2026.gmt"))
    reactome = _parse_gmt_safe(os.path.join(gmt_dir, "Enrichr.Reactome_Pathways_2024.gmt"))

    if not go_bp and not go_cc and not reactome:
        return modules

    pleiotropic_excludes = [
        "positive regulation of cell cycle",
        "negative regulation of cell cycle",
        "regulation of cell cycle (go:0051726)",
        "regulation of cell cycle process",
        "cell cycle arrest",
        "apoptotic",
        "death",
        "meiotic",
        "meiosis",
        "cardiac",
        "neuronal",
        "osteoblast",
        "angiogenesis",
        "t cell",
        "immune",
        "cytokine",
        "insulin",
        "viral",
        "hiv",
    ]
    react_excludes = ["hiv", "viral", "disease"]

    modules["P1_G1S_Origin_Licensing"].update(
        _match_terms(
            go_bp,
            [
                "g1/s transition of mitotic cell cycle",
                "dna-templated dna replication initiation",
                "dna replication initiation",
                "pre-replicative complex",
                "dna unwinding involved in dna replication",
            ],
            pleiotropic_excludes,
        ),
        _match_terms(
            go_cc,
            [
                "mcm complex",
                "nuclear origin of replication recognition complex",
                "replication fork protection complex",
            ],
        ),
        _match_terms(
            reactome,
            [
                "g1/s transition",
                "g1/s-specific transcription",
                "activation of the pre-replicative complex",
                "activation of atr in response to replication stress",
                "orc1 removal from chromatin",
                "assembly of the pre-replicative complex",
                "cdc6",
                "cdt1",
                "e2f mediated regulation of dna replication",
            ],
            react_excludes,
        ),
    )

    modules["P2_S_Replisome_dNTP_Chromatin"].update(
        _match_terms(
            go_bp,
            [
                "dna replication (go:0006260)",
                "dna-templated dna replication (go:0006261)",
                "dna strand elongation",
                "okazaki fragment",
                "deoxyribonucleotide biosynthetic process",
                "deoxyribonucleoside monophosphate",
                "pyrimidine deoxyribonucleotide",
                "nucleosome assembly",
                "chromatin assembly or disassembly",
                "replication-dependent nucleosome",
            ],
            pleiotropic_excludes + ["mitochondrial", "viral"],
        ),
        _match_terms(
            go_cc,
            [
                "replication fork",
                "dna polymerase complex",
                "nuclear replisome",
                "nucleosome (go:0000786)",
                "pcna-p21",
            ],
        ),
        _match_terms(
            reactome,
            [
                "synthesis of dna",
                "dna strand elongation",
                "leading strand synthesis",
                "lagging strand synthesis",
                "processive synthesis on the lagging strand",
                "pcna-dependent",
                "metabolism of nucleotides",
                "interconversion of nucleotide di- and triphosphates",
            ],
            react_excludes,
        ),
    )

    modules["P3_IntraS_G2M_Checkpoint_Repair"].update(
        _match_terms(
            go_bp,
            [
                "dna integrity checkpoint",
                "dna damage checkpoint",
                "intra-s dna damage checkpoint",
                "mitotic g2 dna damage checkpoint",
                "mitotic dna damage checkpoint",
                "double-strand break repair via homologous recombination",
                "recombinational repair",
                "interstrand cross-link repair",
                "replication fork processing",
                "response to replication stress",
            ],
            ["apoptotic", "death", "negative regulation", "positive regulation"],
        ),
        _match_terms(
            reactome,
            [
                "g2/m checkpoints",
                "g2/m dna damage checkpoint",
                "activation of atr in response to replication stress",
                "hdr through homologous recombination",
                "fanconi anemia pathway",
                "resolution of d-loop structures",
            ],
            react_excludes,
        ),
    )

    modules["P4_G2M_Spindle_Kinetochore"].update(
        _match_terms(
            go_bp,
            [
                "g2/m transition of mitotic cell cycle",
                "mitotic spindle organization",
                "mitotic spindle assembly",
                "microtubule cytoskeleton organization involved in mitosis",
                "kinetochore assembly",
                "kinetochore organization",
                "centrosome cycle",
                "centrosome duplication",
                "attachment of spindle microtubules to kinetochore",
                "mitotic nuclear envelope disassembly",
                "positive regulation of ubiquitin protein ligase activity",
            ],
            pleiotropic_excludes,
        ),
        _match_terms(
            go_cc,
            [
                "mitotic spindle",
                "spindle pole",
                "spindle microtubule",
                "kinetochore",
                "condensed chromosome, centromeric region",
                "ndc80 complex",
                "mis12/mind type complex",
                "centrosome",
                "microtubule cytoskeleton",
            ],
        ),
        _match_terms(
            reactome,
            [
                "mitotic g2-g2/m phases",
                "g2/m transition",
                "mitotic prometaphase",
                "resolution of sister chromatid cohesion",
                "recruitment of mitotic centrosome proteins and complexes",
                "polo-like kinase mediated events",
                "cyclin a/b1/b2 associated events during g2/m transition",
                "amplification of signal from the kinetochores",
            ],
            react_excludes,
        ),
    )

    modules["P5_SAC_Segregation_APCC_Cytokinesis"].update(
        _match_terms(
            go_bp,
            [
                "chromosome segregation",
                "sister chromatid segregation",
                "mitotic sister chromatid segregation",
                "chromosome condensation",
                "mitotic chromosome condensation",
                "mitotic spindle assembly checkpoint",
                "spindle assembly checkpoint",
                "mitotic anaphase",
                "anaphase-promoting complex-dependent catabolic process",
                "mitotic cytokinesis",
                "cytokinesis (go:0000910)",
                "contractile ring",
                "midbody",
                "mitotic cell cycle phase transition (go:0044772)",
                "mitotic nuclear division",
            ],
            pleiotropic_excludes,
        ),
        _match_terms(
            go_cc,
            [
                "condensin complex",
                "anaphase-promoting complex",
                "chromosomal passenger complex",
                "midbody",
                "cleavage furrow",
                "centralspindlin complex",
            ],
        ),
        _match_terms(
            reactome,
            [
                "mitotic metaphase and anaphase",
                "mitotic anaphase",
                "mitotic telophase/cytokinesis",
                "separation of sister chromatids",
                "apc/c-mediated degradation of cell cycle proteins",
                "apc/c:cdc20 mediated degradation",
                "apc/c:cdh1 mediated degradation",
                "condensation of prometaphase chromosomes",
                "inactivation of apc/c via direct inhibition of the apc/c complex",
            ],
            react_excludes,
        ),
    )

    return modules


# Canonical Tirosh S and G2M marker sets for computing control-cell phase correlations if obs columns are absent
TIROSH_S_GENES: Set[str] = {
    "MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG", "GINS2", "MCM6",
    "CDCA7", "DTL", "PRIM1", "UHRF1", "MLF1IP", "CENPU", "HELLS", "RFC2", "RPA2", "NASP",
    "RAD51AP1", "GMNN", "WDR76", "SLBP", "CCNE2", "UBR7", "POLD3", "MSH2", "ATAD2", "RAD51",
    "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1", "BLM", "CASP8AP2", "USP1", "CLSPN",
    "POLA1", "CHAF1B", "BRIP1", "E2F8",
}

TIROSH_G2M_GENES: Set[str] = {
    "HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80", "CKS2", "NUF2",
    "CKS1B", "MKI67", "TMPO", "CENPF", "TACC3", "FAM64A", "PIMREG", "SMC4", "CCNB2", "CKAP2L",
    "CKAP2", "AURKB", "BUB1", "KIF11", "ANP32E", "TUBB4B", "GTSE1", "KIF20B", "HJURP", "CDCA3",
    "HN1", "JPT1", "CDC20", "TTK", "CDC25C", "KIF2C", "RANGAP1", "NCAPD2", "DLGAP5", "CDCA2",
    "CDCA8", "ECT2", "KIF23", "HMMR", "AURKA", "PSRC1", "ANLN", "LBR", "CKAP5", "CENPE",
    "CTCF", "NEK2", "G2E3", "GAS2L3", "CBX5", "CENPA",
}


def build_go_cycle_and_quarantine_masks(
    adata,
    cycle_gene_mask_key: str = "is_cycle_gene",
    control_cell_mask: Optional[np.ndarray] = None,
    layer: Optional[str] = "counts",
    r_pos_go_thresh: float = 0.20,
    r_pos_core_thresh: float = 0.30,
    r_pos_ood_thresh: float = 0.20,
    gmt_dir: str = "/home/sihan/.cache/gseapy",
) -> Dict[str, Union[torch.Tensor, np.ndarray, pd.DataFrame, List[str]]]:
    """Constructs the Dual-Gate Structural GO/Reactome Cycle & Quarantine Masks.

    Returns a dictionary containing:
      - ``tirosh_mask``: (G,) bool tensor of the canonical Tirosh cycle genes (e.g. 68 in CD8+).
      - ``cycle_hard_mask``: (G,) bool tensor of G_{cyc,hard} (e.g. 249 in CD8+: 68 Tirosh + 181 GO/core).
      - ``ood_quarantine_mask``: (G,) bool tensor of G_{ood,quar} (e.g. 400 in CD8+: 68 Tirosh + 332 mitotic).
      - ``pathway_mask``: (6, G) float32 tensor assigning genes in G_{cyc,hard} to K=6 GO modules P1..P6.
      - ``pathway_names``: list of 6 module names.
      - ``gene_audit_df``: DataFrame of per-gene correlations and mask memberships.
    """
    var_names = np.asarray(adata.var_names, dtype=str)
    G = len(var_names)

    if cycle_gene_mask_key in adata.var.columns:
        tirosh_bool = np.asarray(adata.var[cycle_gene_mask_key].values, dtype=bool)
    else:
        tirosh_all = TIROSH_S_GENES | TIROSH_G2M_GENES
        tirosh_bool = np.array([g in tirosh_all for g in var_names], dtype=bool)

    # Extract expression matrix on control cells (or all cells if control mask has <20 cells)
    if control_cell_mask is not None and int(np.sum(control_cell_mask)) >= 20:
        cell_sel = np.asarray(control_cell_mask, dtype=bool)
    elif "is_treated" in adata.obs.columns and int(np.sum(adata.obs["is_treated"].values < 0.5)) >= 20:
        cell_sel = np.asarray(adata.obs["is_treated"].values < 0.5, dtype=bool)
    else:
        cell_sel = np.ones(adata.n_obs, dtype=bool)

    if layer is not None and layer in adata.layers:
        X_sub = adata.layers[layer][cell_sel]
    else:
        X_sub = adata.X[cell_sel]

    X_dense = X_sub.toarray().astype(np.float32) if sp.issparse(X_sub) else np.asarray(X_sub, dtype=np.float32)
    X_nonneg = np.maximum(X_dense, 0.0)
    tot_counts = np.maximum(X_nonneg.sum(axis=1, keepdims=True), 1.0)
    X_log = np.log1p(X_nonneg * (1e4 / tot_counts))

    # Obtain S_score and G2M_score on control cells
    if "S_score" in adata.obs.columns and "G2M_score" in adata.obs.columns:
        s_sc = np.asarray(adata.obs["S_score"].values[cell_sel], dtype=np.float64)
        g2m_sc = np.asarray(adata.obs["G2M_score"].values[cell_sel], dtype=np.float64)
    else:
        s_idx = np.array([g in TIROSH_S_GENES for g in var_names], dtype=bool)
        g2m_idx = np.array([g in TIROSH_G2M_GENES for g in var_names], dtype=bool)
        bg_mean = X_log.mean(axis=1)
        s_sc = (X_log[:, s_idx].mean(axis=1) - bg_mean).astype(np.float64) if s_idx.any() else np.zeros(X_log.shape[0])
        g2m_sc = (X_log[:, g2m_idx].mean(axis=1) - bg_mean).astype(np.float64) if g2m_idx.any() else np.zeros(X_log.shape[0])

    # Compute signed Pearson correlations r_S^ctrl(g) and r_G2M^ctrl(g)
    xc = X_log - X_log.mean(axis=0, keepdims=True)
    x_std = np.maximum(xc.std(axis=0), 1e-12)
    s_std = max(float(s_sc.std()), 1e-12)
    g2m_std = max(float(g2m_sc.std()), 1e-12)
    s_z = (s_sc - s_sc.mean()) / s_std
    g2m_z = (g2m_sc - g2m_sc.mean()) / g2m_std

    n_cells = float(X_log.shape[0])
    r_s = ((xc.T @ s_z) / (n_cells * x_std)).astype(np.float32)
    r_g2m = ((xc.T @ g2m_z) / (n_cells * x_std)).astype(np.float32)

    r_pos = np.maximum(r_s, r_g2m)
    r_neg = np.minimum(r_s, r_g2m)
    pos_dominant = r_pos > np.abs(r_neg)

    # Load structural GO/Reactome modules P1..P5
    modules_dict = load_structural_go_modules(gmt_dir=gmt_dir)
    p1_set = modules_dict["P1_G1S_Origin_Licensing"]
    p2_set = modules_dict["P2_S_Replisome_dNTP_Chromatin"]
    p3_set = modules_dict["P3_IntraS_G2M_Checkpoint_Repair"]
    p4_set = modules_dict["P4_G2M_Spindle_Kinetochore"]
    p5_set = modules_dict["P5_SAC_Segregation_APCC_Cytokinesis"]
    struct5_union = p1_set | p2_set | p3_set | p4_set | p5_set

    in_struct5 = np.array([g in struct5_union for g in var_names], dtype=bool)
    in_protect = np.array([g in PROTECTED_PLEIOTROPIC_GENES for g in var_names], dtype=bool)

    # 1. Hard Cycle Mask G_{cyc,hard}:
    #    Tirosh U (([P1..P5 & r_pos > r_pos_go_thresh] U [r_pos > r_pos_core_thresh]) & pos_dominant \ G_Protect)
    non_tirosh_go_gate = (~tirosh_bool) & (~in_protect) & pos_dominant & (
        (in_struct5 & (r_pos > float(r_pos_go_thresh))) | (r_pos > float(r_pos_core_thresh))
    )
    cycle_hard_bool = tirosh_bool | non_tirosh_go_gate

    # 2. OOD Mitotic Quarantine Mask G_{ood,quar}:
    #    G_{cyc,hard} U ({r_pos > r_pos_ood_thresh & pos_dominant} \ G_Protect)
    ood_quar_bool = cycle_hard_bool | (
        (~tirosh_bool) & (~in_protect) & pos_dominant & (r_pos > float(r_pos_ood_thresh))
    )

    # 3. 6-Module GO Pathway Assignment Matrix M_path in {0, 1}^{6 x G}
    M_path = np.zeros((6, G), dtype=np.float32)
    module_sets_5 = [p1_set, p2_set, p3_set, p4_set, p5_set]
    for g_idx, g_name in enumerate(var_names):
        if not cycle_hard_bool[g_idx]:
            continue
        assigned_any = False
        for k_idx, m_set in enumerate(module_sets_5):
            if g_name in m_set:
                M_path[k_idx, g_idx] = 1.0
                assigned_any = True
        # Module 6 (P6_Core_Mitotic_Execution_Residual) receives all residual cycle_hard genes
        # plus core high-amplitude mitotic markers (r_pos > 0.30 or Tirosh) so P6 anchors global mitotic amplitude
        if (not assigned_any) or (r_pos[g_idx] > float(r_pos_core_thresh)) or tirosh_bool[g_idx]:
            M_path[5, g_idx] = 1.0

    audit_df = pd.DataFrame(
        {
            "gene": var_names,
            "is_tirosh": tirosh_bool,
            "in_struct5_go": in_struct5,
            "in_protect": in_protect,
            "r_S_ctrl": r_s,
            "r_G2M_ctrl": r_g2m,
            "r_pos_ctrl": r_pos,
            "r_neg_ctrl": r_neg,
            "pos_dominant": pos_dominant,
            "in_cycle_hard_mask": cycle_hard_bool,
            "in_ood_quarantine_mask": ood_quar_bool,
            "P1_G1S": M_path[0].astype(bool),
            "P2_S_Rep": M_path[1].astype(bool),
            "P3_Chk": M_path[2].astype(bool),
            "P4_G2M": M_path[3].astype(bool),
            "P5_Seg": M_path[4].astype(bool),
            "P6_CoreRes": M_path[5].astype(bool),
        }
    )

    return {
        "tirosh_mask": torch.tensor(tirosh_bool, dtype=torch.bool),
        "cycle_hard_mask": torch.tensor(cycle_hard_bool, dtype=torch.bool),
        "ood_quarantine_mask": torch.tensor(ood_quar_bool, dtype=torch.bool),
        "pathway_mask": torch.tensor(M_path, dtype=torch.float32),
        "pathway_names": list(GO_PATHWAY_NAMES),
        "gene_audit_df": audit_df,
    }
