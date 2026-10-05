import DNA_Markov_Model as dmm
 
rate_constant = dmm.strand_displacement_rate(
    target="CTATCTCTGTCTGTCTCTTCTATCTCTTGTGTCTCTG",
    incumbent="AGAGATAGAAGAGACAGACAGAGATAG",
    invader="CAGAGACACAAGACATAGAAGAGACAGACAGAGATAG",
    temperature=25.0,
    na_conc=0.5,
    mg_conc=0.0,
    c_oligo=10e-9,
)

print(f"Rate constant: {rate_constant:.1e} /s")