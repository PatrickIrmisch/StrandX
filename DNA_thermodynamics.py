# Import & settings
import numpy as np
import SantaLucia_DNA_parameters as NNparams

# Direct references to parameter dictionaries
data_DNA = NNparams.data_DNA_NN4
data_DNA_internalMM = NNparams.data_DNA_IMM1
data_DNA_terminal = NNparams.data_DNA_TMM1
data_DNA_dangle = NNparams.data_DNA_DE1

# Shared Watson-Crick base pair set
WATSON_CRICK_PAIRS = {("A", "T"), ("T", "A"), ("G", "C"), ("C", "G")}


def _lookup_stack(key, key_alt, dict_priority):
    """
    Look up (dH, dS) tuple given primary key, alternative key, 
    and a sequence of dictionaries in priority order.
    """
    for data_dict in dict_priority:
        if key in data_dict:
            return data_dict[key], key
        if key_alt in data_dict:
            return data_dict[key_alt], key_alt
    return None, None


def base_pairing_energies(
    DNA_Sequence_1,
    DNA_sequence_2,
    Na_conc,
    Mg_conc,
    Temperature,
    oligo_concentration,
    output_stacks=False,
):
    """
    Calculate base pairing energies using SantaLucia Unified NN Parameters.
    dH: kcal/mol, dS: cal/mol·K, dG: kcal/mol.
    """
    if len(DNA_Sequence_1) != len(DNA_sequence_2):
        raise ValueError("Sequences must have the same length")

    n = len(DNA_Sequence_1)
    if n < 5:
        raise ValueError("Sequence must be at least 4 bases long.")

    T_kelvin = Temperature + 273.15
    total_H = 0.0  # in kcal/mol
    total_S = 0.0  # in cal/mol·K

    # 1. Monovalent Equivalent for Salt Correction (Owczarzy et al. 2008)
    monovalent_eq = Na_conc + (3.3 * np.sqrt(Mg_conc))
    salt_correction_per_link = 0.368 * np.log(monovalent_eq)

    # 2. Initiation penalty (Salt-independent)
    ini_H, ini_S = data_DNA["init"]
    total_H += ini_H
    total_S += ini_S

    if output_stacks:
        print("Init:", ini_H, ini_S)

    # Helper for terminal A/T initiation penalties
    def _apply_at_penalty(terminal_base_pair):
        nonlocal total_H, total_S
        if terminal_base_pair in ("AT", "TA"):
            at_h, at_s = data_DNA["init_A/T"]
            total_H += at_h
            total_S += at_s
            if output_stacks:
                print(f"Terminal AT penalty: {at_h} {at_s}")

    # 3. Left End Stacking
    dinuc_L1 = DNA_Sequence_1[0:2]
    dinuc_L2 = DNA_sequence_2[0:2]
    left_key = f"{dinuc_L1}/{dinuc_L2}"
    left_key_alt = f"{dinuc_L2[::-1]}/{dinuc_L1[::-1]}"

    # Specific precedence order for the left end
    left_dicts = [
        data_DNA,
        data_DNA_internalMM,
        data_DNA_terminal,
        data_DNA_dangle,
    ]
    res, matched_key = _lookup_stack(left_key, left_key_alt, left_dicts)

    if res is None:
        raise ValueError(f"Error on left end stack: {left_key}")

    H, S = res
    total_H += H
    total_S += S + salt_correction_per_link

    # Check for terminal A/T penalty on standard matches/mismatches
    if matched_key in (left_key, left_key_alt) and res in data_DNA.values():
        left_end = DNA_Sequence_1[0] + DNA_sequence_2[0]
        _apply_at_penalty(left_end)

    if output_stacks:
        print(left_key, H, S)

    # 4. Internal Stacks
    internal_dicts = [data_DNA, data_DNA_internalMM]
    for i in range(1, n - 2):
        dinuc_1 = DNA_Sequence_1[i : i + 2]
        dinuc_2 = DNA_sequence_2[i : i + 2]
        key = f"{dinuc_1}/{dinuc_2}"
        key_alt = f"{dinuc_2[::-1]}/{dinuc_1[::-1]}"

        res, _ = _lookup_stack(key, key_alt, internal_dicts)
        if res is None:
            raise ValueError(f"Error! Two mismatches next to each other: {key}")

        H, S = res
        total_H += H
        total_S += S + salt_correction_per_link

        if output_stacks:
            print(key, H, S)

    # 5. Right End Stacking
    dinuc_R1 = DNA_Sequence_1[-2:]
    dinuc_R2 = DNA_sequence_2[-2:]
    right_key = f"{dinuc_R1}/{dinuc_R2}"
    right_key_alt = f"{dinuc_R2[::-1]}/{dinuc_R1[::-1]}"

    # Specific precedence order for the right end
    right_dicts = [
        data_DNA,
        data_DNA_terminal,
        data_DNA_internalMM,
        data_DNA_dangle,
    ]
    res, matched_key = _lookup_stack(right_key, right_key_alt, right_dicts)

    if res is None:
        raise ValueError(f"Error on right end stack: {right_key}")

    H, S = res
    total_H += H
    total_S += S + salt_correction_per_link

    # Check for terminal A/T penalty on standard matches/mismatches
    if matched_key in (right_key, right_key_alt) and res in data_DNA.values():
        right_end = DNA_Sequence_1[-1] + DNA_sequence_2[-1]
        _apply_at_penalty(right_end)

    if output_stacks:
        print(right_key, H, S)

    # 6. Final Gibbs Free Energy Calculation
    full_G = total_H * 1000 - (T_kelvin * total_S)

    return total_H, total_S, full_G / 1000


def find_best_duplex_alignment(seq1, seq2):
    """
    Find the best rigid duplex alignment between two DNA strands.
    """
    comp = str.maketrans("ATGC", "TACG")
    rc2 = seq2.translate(comp)[::-1]

    best = None

    for offset in range(-len(rc2) + 1, len(seq1)):
        i_start = max(0, offset)
        i_end = min(len(seq1), offset + len(rc2))

        if i_start >= i_end:
            continue

        j_start = i_start - offset
        j_end = j_start + (i_end - i_start)

        paired1 = seq1[i_start:i_end]
        paired2_rc = rc2[j_start:j_end]

        mismatch_positions = [
            i for i, (a, b) in enumerate(zip(paired1, paired2_rc)) if a != b
        ]
        correct_base_at_mismatches = [
            paired2_rc[i] for i in mismatch_positions
        ]

        seq2_start = len(seq2) - j_end
        seq2_end = len(seq2) - j_start

        candidate = {
            "seq1": seq1,
            "seq2": seq2,
            "seq1_start": i_start,
            "seq1_end": i_end,
            "seq2_start": seq2_start,
            "seq2_end": seq2_end,
            "offset": offset,
            "overlap": len(paired1),
            "mismatches": len(mismatch_positions),
            "mismatch_positions": mismatch_positions,
            "correct_base_at_mismatches": correct_base_at_mismatches,
        }

        candidate["score"] = candidate["overlap"] - candidate["mismatches"]

        if (
            best is None
            or candidate["score"] > best["score"]
            or (
                candidate["score"] == best["score"]
                and candidate["overlap"] > best["overlap"]
            )
        ):
            best = candidate

    if best is None:
        raise ValueError("No valid duplex alignment")

    return best


def print_duplex_alignment(aln):
    """
    Format and print duplex alignment for visual inspection.
    """
    seq1, seq2 = aln["seq1"], aln["seq2"]
    s1_start, s1_end = aln["seq1_start"], aln["seq1_end"]
    s2_start, s2_end = aln["seq2_start"], aln["seq2_end"]

    left = max(s1_start, len(seq2) - s2_end)
    right = max(len(seq1) - s1_end, s2_start)
    paired_len = s1_end - s1_start

    L = left + paired_len + right

    top = [" "] * L
    bottom = [" "] * L
    match = [" "] * L

    # Top strand placement
    pos = left - s1_start
    for b in seq1:
        top[pos] = b
        pos += 1

    # Bottom strand placement (antiparallel)
    seq2_display = seq2[::-1]
    display_s2_start = len(seq2) - s2_end
    pos = left - display_s2_start
    for b in seq2_display:
        bottom[pos] = b
        pos += 1

    # Pairing symbols
    for i in range(paired_len):
        t = left + i
        b = left + i
        if (top[t], bottom[b]) in WATSON_CRICK_PAIRS:
            match[t] = "|"
        else:
            match[t] = "x"

    return (
        f"5'- {''.join(top)} -3'\n"
        f"    {''.join(match)}\n"
        f"3'- {''.join(bottom)} -5'"
    )


def crop_duplex_for_nn(aln):
    """
    Crop duplex alignment for nearest-neighbor parameter lookup.
    Retains terminal dangling bases adjacent to Watson-Crick pairs.
    """
    seq1, seq2 = aln["seq1"], aln["seq2"]
    s1_start, s1_end = aln["seq1_start"], aln["seq1_end"]
    s2_start, s2_end = aln["seq2_start"], aln["seq2_end"]

    top_out = list(seq1[s1_start:s1_end])
    bottom_out = list(seq2[s2_start:s2_end][::-1])

    # Left dangling end
    if s1_start > 0 or s2_end < len(seq2):
        if (top_out[0], bottom_out[0]) in WATSON_CRICK_PAIRS:
            if s1_start > 0:
                top_out.insert(0, seq1[s1_start - 1])
                bottom_out.insert(0, ".")
            elif s2_end < len(seq2):
                top_out.insert(0, ".")
                bottom_out.insert(0, seq2[s2_end])

    # Right dangling end
    if s1_end < len(seq1) or s2_start > 0:
        if (top_out[-1], bottom_out[-1]) in WATSON_CRICK_PAIRS:
            if s1_end < len(seq1):
                top_out.append(seq1[s1_end])
                bottom_out.append(".")
            elif s2_start > 0:
                top_out.append(".")
                bottom_out.append(seq2[s2_start - 1])

    return "".join(top_out), "".join(bottom_out)


def identify_domains(aln_inc, aln_inv, target):
    """
    Compare target-incumbent and target-invader alignments to identify 
    displacement and toehold regions.
    """
    inc_start, inc_end = aln_inc["seq2_start"], aln_inc["seq2_end"]
    inv_start, inv_end = aln_inv["seq2_start"], aln_inv["seq2_end"]

    disp_start = max(inc_start, inv_start)
    disp_end = min(inc_end, inv_end)

    left_start = inv_start
    left_end = min(inv_end, inc_start)

    right_start = max(inv_start, inc_end)
    right_end = inv_end

    return {
        "displacement_start": disp_start,
        "displacement_end": disp_end,
        "displacement_sequence": target[disp_start:disp_end],
        "toehold_left": target[left_start:left_end],
        "toehold_right": target[right_start:right_end],
    }