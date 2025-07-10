TARGETSCAN_COLUMN_TO_SEQUENCE = {
    'hsa-miR-16-5p': 'TAGCAGCACGTAAATATTGGCG', 
    'hsa-miR-106b-5p': 'TAAAGTGCTGACAGTGCAGAT', 
    # 'hsa-miR-200a-3p': 'TAACACTGUCTGGTAACGATGT', # this was an old error
    'hsa-miR-200a-3p': 'TAACACTGTCTGGTAACGATGT', # https://mirbase.org/hairpin/MI0000737
    'hsa-miR-200b-3p': 'TAATACTGCCTGGTAATGATGA', 
    'hsa-miR-215-5p': 'ATGACCTATGAATTGACAGAC', 
    'hsa-let-7c-5p': 'TGAGGTAGTAGGTTGTATGGTT', 
    'hsa-miR-103a-3p': 'AGCAGCATTGTACAGGGCTATGA'
}

PSILAC_DATA = {
    'log2FC median let7': {
        'sequence': 'TGGAAGACTAGTGATTTTGTTGTT',  # hsa-let-7c-5p
        'name': 'hsa-let-7c-5p',  # https://mirbase.org/hairpin/MI0000263
    },
    'log2FC median miR1': {
        'sequence': 'ACATACCTCTTTATATGCCCAT',  # hsa-miR-1-5p
        'name': 'hsa-miR-1-5p',
    },
    'log2FC median miR155': {
        'sequence': 'TTAATGCTAATCGTGATAGGGGTT',  # hsa-Mir-155_5p
        'name': 'hsa-Mir-155_5p',
    },
    'log2FC median miR16': {
        'sequence': 'TAGCAGCACGTAAATATTGGCG',  # hsa-miR-16-5p
        'name': 'hsa-miR-16-5p',
    },
    'log2FC median miR30': {
        'sequence': 'TGTAAACATCCTCGACTGGAAG',  # hsa-miR-30a-5p
        'name': 'hsa-miR-30a-5p',  # https://mirbase.org/hairpin/MI0000088
    },
}


HELA_TRANSFACTION_DATA = {
    'let7': {
        'sequence': 'TGGAAGACTAGTGATTTTGTTGTT',  # hsa-let-7c-5p
        'name': 'hsa-let-7c-5p',
    },
    'lsy6': {
        'sequence': '',
        'name': '',
    },
    'mir1': {
        'sequence': 'ACATACCTCTTTATATGCCCAT',  # hsa-miR-1-5p
        'name': 'hsa-miR-1-5p',
    },
    'mir124': {
        'sequence': 'CGTGTTCACAGCGGACCTTGAT',
        'name': 'hsa-miR-124-5p',  # https://mirbase.org/hairpin/MI0000443
    },
    'mir137': {
        'sequence': 'ACGGGTATTCTTGGGTGGATAAT',
        'name': 'hsa-miR-137-5p',  # https://mirbase.org/hairpin/MI0000454
    },
    'mir139': {
        'sequence': 'TCTACAGTGCACGTGTCTCCAGT',
        'name': 'hsa-miR-139-5p',  # https://www.mirbase.org/mature/MIMAT0000250
    },
    'mir143': {
        'sequence': 'GGTGCAGTGCTGCATCTCCTGG',
        'name': 'hsa-miR-143-5p',
    },
    'mir144': {
        'sequence': '',
        'name': '',
    },
    'mir153': {
        'sequence': '',
        'name': '',
    },
    'mir155': {
        'sequence': 'TTAATGCTAATCGTGATAGGGGTT',  # hsa-Mir-155_5p
        'name': 'hsa-Mir-155_5p',
    },
    'mir182': {
        'sequence': '',
        'name': '',
    },
    'mir199a': {
        'sequence': '',
        'name': '',
    },
    'mir204': {
        'sequence': '',
        'name': '',
    },
    'mir205': {
        'sequence': '',
        'name': '',
    },
    'mir216b': {
        'sequence': '',
        'name': '',
    },
    'mir223': {
        'sequence': '',
        'name': '',
    },
    'mir7': {
        'sequence': '',
        'name': '',
    },
}
