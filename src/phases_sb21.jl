#
# This file defines CONSTANTS that are used throughout the project:
# PP, SS, PP_COMP, SS_COMP, IDX_of_variable_components_in_SS
#

PP = ["qtz", "coe", "st", "ky", "neph", "capv", "co"];  # add "aMgO", "aFeO", "aAl2O3" ?
SS = ["plg", "sp", "ol", "wa", "ri", "opx", "cpx", "hpcpx", "ak", "gtmj", "pv", "ppv", "cf", "mw", "nal"];

# CONSTANT used within the pre-processing functions
# Some phases are never stable for the P–T–BULK conditions considered:
# Corundum, Post-Perovskite
# these should not be considered by the surrogate.
IDX_OF_PHASES_NEVER_STABLE = [7, 19]
IDX_PP_NEVER_STABLE = IDX_OF_PHASES_NEVER_STABLE[IDX_OF_PHASES_NEVER_STABLE .<= 7]
IDX_SS_NEVER_STABLE = IDX_OF_PHASES_NEVER_STABLE[IDX_OF_PHASES_NEVER_STABLE .> 7] .-7

# --------------------------------------------------------------------
# PP composition in molar fraction of oxides
# following "Xoxides = ["SiO2"; "CaO"; "Al2O3"; "FeO"; "MgO"; "Na2O"]"
qtz  = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
coe  = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
st   = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
ky   = [0.5, 0.0, 0.5, 0.0, 0.0, 0.0];
neph = [0.5, 0.0, 0.25, 0.0, 0.0, 0.25];
capv = [0.5, 0.5, 0.0, 0.0, 0.0, 0.0];
co   = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0];

PP_COMP = Float32.(vcat(eval.(Symbol.(PP))...));
PP_COMP_adj = Float32.(vcat(eval.(Symbol.(PP[[i for i in 1:7 if i ∉ IDX_PP_NEVER_STABLE]]))...));

# --------------------------------------------------------------------
# SS composition in molar fraction of oxides
# following "Xoxides = ["SiO2"; "CaO"; "Al2O3"; "FeO"; "MgO"; "Na2O"]"

# extract the component in each SS that is neither zero nor constant,
# this is the only compositional variable that must be predicted.
# All other molar fractions of the invariant components can be inserted
# into the SS composition post prediction.
idx_of_variable_components_in_SS = [];
idx_of_constant_components_in_SS = [];

function find_variable_components(em_comp)
    em_comp_mat = hcat(em_comp...)
    return findall(row -> length(unique(row)) > 1, eachrow(em_comp_mat))
end
function find_constant_components(em_comp)
    em_comp_mat = hcat(em_comp...)
    return findall(row -> length(unique(row)) == 1, eachrow(em_comp_mat))
end

# SS phases and their end-members
# Plagioclase
# anorthite "Ca_1Al_2Si_2O_8"
an = [2.0, 1.0, 1.0, 0.0, 0.0, 0.0];
an ./= sum(an);
# albite "Na_1Al_1Si_3O_8"
ab = [3., 0.0, 0.5, 0.0, 0.0, 0.5];
ab ./= sum(ab);
push!(idx_of_variable_components_in_SS, find_variable_components([an, ab]));
push!(idx_of_constant_components_in_SS, find_constant_components([an, ab]));

plg = zeros(6);
plg[idx_of_constant_components_in_SS[end]] .= an[idx_of_constant_components_in_SS[end]];

# Spinel
# spinel "(Mg_3Al_1)(Al_7Mg_1)O_16"
spi = [0.0, 0.0, 4.0, 0.0, 4.0, 0.0];
spi ./= sum(spi);
# hercynite "(Fe_3Al_1)(Al_7Fe_1)O_16"
hc = [0.0, 0.0, 4.0, 4.0, 0.0, 0.0];
hc ./= sum(hc);
push!(idx_of_variable_components_in_SS, find_variable_components([spi, hc]));
push!(idx_of_constant_components_in_SS, find_constant_components([spi, hc]));

sp = zeros(6);
sp[idx_of_constant_components_in_SS[end]] .= spi[idx_of_constant_components_in_SS[end]];

# Olivine
# forsterite "Mg_2Si_1O_4"
fo = [1.0, 0.0, 0.0, 0.0, 2.0, 0.0];
fo ./= sum(fo);
# fayalite "Fe_2Si_1O_4"
fa = [1.0, 0.0, 0.0, 2.0, 0.0, 0.0];
fa ./= sum(fa);
push!(idx_of_variable_components_in_SS, find_variable_components([fo, fa]));
push!(idx_of_constant_components_in_SS, find_constant_components([fo, fa]));

ol = zeros(6);
ol[idx_of_constant_components_in_SS[end]] .= fo[idx_of_constant_components_in_SS[end]];

# Wadsleyite
# mg-wadsleyite "Mg_2Si_1O_4"
mgwa = [1.0, 0.0, 0.0, 0.0, 2.0, 0.0];
mgwa ./= sum(mgwa);
# fe-wadsleyite "Fe_2Si_1O_4"
fewa = [1.0, 0.0, 0.0, 2.0, 0.0, 0.0];
fewa ./= sum(fewa);
push!(idx_of_variable_components_in_SS, find_variable_components([mgwa, fewa]));
push!(idx_of_constant_components_in_SS, find_constant_components([mgwa, fewa]));

wa = zeros(6);
wa[idx_of_constant_components_in_SS[end]] .= mgwa[idx_of_constant_components_in_SS[end]];

# Ringwoodite
# mg-ringwoodite "Mg_2Si_1O_4",
mgri = [1.0, 0.0, 0.0, 0.0, 2.0, 0.0];
mgri ./= sum(mgri);
# fe-ringwoodite "Fe_2Si_1O_4"
feri = [1.0, 0.0, 0.0, 2.0, 0.0, 0.0];
feri ./= sum(feri);
push!(idx_of_variable_components_in_SS, find_variable_components([mgri, feri]));
push!(idx_of_constant_components_in_SS, find_constant_components([mgri, feri]));

ri = zeros(6);
ri[idx_of_constant_components_in_SS[end]] .= mgri[idx_of_constant_components_in_SS[end]];

# Orthopyroxene
# enstatite "Mg_1Mg_1Si_2O_6"
en = [2.0, 0.0, 0.0, 0.0, 2.0, 0.0];
en ./= sum(en);
# ferrosilite "Fe_1Fe_1Si_2O_6"
fs = [2.0, 0.0, 0.0, 2.0, 0.0, 0.0];
fs ./= sum(fs);
# mg-tschermak "Mg_1Al_1Si_1O_6Al_1"
mgts = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
mgts ./= sum(mgts);
# ortho-diopside "Ca_1Mg_1Si_2O_6"
odi = [2.0, 1.0, 0.0, 0.0, 1.0, 0.0];
odi ./= sum(odi);
push!(idx_of_variable_components_in_SS, find_variable_components([en, fs, mgts, odi]));
push!(idx_of_constant_components_in_SS, find_constant_components([en, fs, mgts, odi]));

opx = zeros(6);
opx[idx_of_constant_components_in_SS[end]] .= en[idx_of_constant_components_in_SS[end]];

# Clinopyroxene
# diopside "Ca_1Mg_1Si_2O_6"
di = [2.0, 1.0, 0.0, 0.0, 1.0, 0.0];
di ./= sum(di);
# hedenbergite "Ca_1Fe_1Si_2O_6"
he = [2.0, 1.0, 0.0, 1.0, 0.0, 0.0];
he ./= sum(he);
# clinoenstatite "Mg_1Mg_1Si_2O_6"
cen = [2.0, 0.0, 0.0, 0.0, 2.0, 0.0];
cen ./= sum(cen);
# ca-tschermak "Ca_1Al_1(Si_1Al_1)O_6"
cats = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
cats ./= sum(cats);
# jadeite "Na_1Al_1Si_2O_6"
jd = [2.0, 0.0, 0.5, 0.0, 0.0, 0.5];
jd ./= sum(jd);
push!(idx_of_variable_components_in_SS, find_variable_components([di, he, cen, cats, jd]));
push!(idx_of_constant_components_in_SS, find_constant_components([di, he, cen, cats, jd]));

cpx = zeros(6);
cpx[idx_of_constant_components_in_SS[end]] .= di[idx_of_constant_components_in_SS[end]];

# HP-clinopyroxene
# hp-clinoenstatite "Mg_2Si_2O_6"
hpcen = [2.0, 0.0, 0.0, 0.0, 2.0, 0.0];
hpcen ./= sum(hpcen);
# hp-clinoferrosilite "Fe_2Si_2O_6"
hpcfs = [2.0, 0.0, 0.0, 2.0, 0.0, 0.0];
hpcfs ./= sum(hpcfs);
push!(idx_of_variable_components_in_SS, find_variable_components([hpcen, hpcfs]));
push!(idx_of_constant_components_in_SS, find_constant_components([hpcen, hpcfs]));

hpcpx = zeros(6);
hpcpx[idx_of_constant_components_in_SS[end]] .= hpcen[idx_of_constant_components_in_SS[end]];

# Akimotoite
# mg-akimotoite "Mg_1Si_1O_3"
mgak = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
mgak ./= sum(mgak);
# fe-akimotoite "Fe_1Si_1O_3"
feak = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
feak ./= sum(feak);
# corundum "Al_2O_3"
cor = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
cor ./= sum(cor);
push!(idx_of_variable_components_in_SS, find_variable_components([mgak, feak, cor]));
push!(idx_of_constant_components_in_SS, find_constant_components([mgak, feak, cor]));

ak = zeros(6);
ak[idx_of_constant_components_in_SS[end]] .= mgak[idx_of_constant_components_in_SS[end]];

# Garnet-Majorite
# pyrope "Mg_3Al_1Al_1Si_3O_12"
py = [3.0, 0.0, 1.0, 0.0, 3.0, 0.0];
py ./= sum(py);
# almandine "Fe_3Al_1Al_1Si_3O_12"
al = [3.0, 0.0, 1.0, 3.0, 0.0, 0.0];
al ./= sum(al);
# grossular "Ca_3Al_1Al_1Si_3O_12"
gr = [3.0, 3.0, 1.0, 0.0, 0.0, 0.0];
gr ./= sum(gr);
# mg-majorite "Mg_3Mg_1Si_1Si_3O_12"
mgmj = [4.0, 0.0, 0.0, 0.0, 4.0, 0.0];
mgmj ./= sum(mgmj);
# jd-majorite "(Na_2Mg_1)Si_1Si_1Si_3O_12"
namj = [5.0, 0.0, 0.0, 0.0, 1.0, 1.0];
namj ./= sum(namj);
push!(idx_of_variable_components_in_SS, find_variable_components([py, al, gr, mgmj, namj]));
push!(idx_of_constant_components_in_SS, find_constant_components([py, al, gr, mgmj, namj]));

gtmj = zeros(6);
gtmj[idx_of_constant_components_in_SS[end]] .= py[idx_of_constant_components_in_SS[end]];

# Perovskite (Bridgmanite)
# mg-perovskite "Mg_1Si_1O_3"
mgbg = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
mgbg ./= sum(mgbg);
# fe-perovskite "Fe_1Si_1O_3"
febg = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
febg ./= sum(febg);
# al-perovskite "Al_1Al_1O_3"
albg = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
albg ./= sum(albg);
push!(idx_of_variable_components_in_SS, find_variable_components([mgbg, febg, albg]));
push!(idx_of_constant_components_in_SS, find_constant_components([mgbg, febg, albg]));

pv = zeros(6);
pv[idx_of_constant_components_in_SS[end]] .= mgbg[idx_of_constant_components_in_SS[end]];

# Post-perovskite
# mg-post-perovskite "Mg_1Si_1O_3"
mppv = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
mppv ./= sum(mppv);
# fe-post-perovskite", "Fe_1Si_1O_3"
fppv = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
fppv ./= sum(fppv);
# al-post-perovskite "Al_1Al_1O_3"
appv = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
appv ./= sum(appv);
push!(idx_of_variable_components_in_SS, find_variable_components([mppv, fppv, appv]));
push!(idx_of_constant_components_in_SS, find_constant_components([mppv, fppv, appv]));

ppv = zeros(6);
ppv[idx_of_constant_components_in_SS[end]] .= mppv[idx_of_constant_components_in_SS[end]];

# Ca-ferrite
# mg-ca-ferrite "Mg_1Al_1Al_1O_4"
mgcf = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0];
mgcf ./= sum(mgcf);
# fe-ca-ferrite "Fe_1Al_1Al_1O_4"
fecf = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
fecf ./= sum(fecf);
# na-ca-ferrite "Na_1Al_1Si_1O_4"
nacf = [1.0, 0.0, 0.5, 0.0, 0.0, 0.5];
nacf ./= sum(nacf);
push!(idx_of_variable_components_in_SS, find_variable_components([mgcf, fecf, nacf]));
push!(idx_of_constant_components_in_SS, find_constant_components([mgcf, fecf, nacf]));

cf = zeros(6);
cf[idx_of_constant_components_in_SS[end]] .= mgcf[idx_of_constant_components_in_SS[end]];

# Magnesiowüstite (Ferropericlase)
# periclase "Mg_2Mg_2O_4"
pe = [0.0, 0.0, 0.0, 0.0, 4.0, 0.0];
pe ./= sum(pe);
# wustite "Fe_2Fe_2O_4"
wu = [0.0, 0.0, 0.0, 4.0, 0.0, 0.0];
wu ./= sum(wu);
#  alpha-nao2-phase "Na_2Al_2O_4"
anao = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
anao ./= sum(anao);
push!(idx_of_variable_components_in_SS, find_variable_components([pe, wu, anao]));
push!(idx_of_constant_components_in_SS, find_constant_components([pe, wu, anao]));

mw = zeros(6);
mw[idx_of_constant_components_in_SS[end]] .= pe[idx_of_constant_components_in_SS[end]];

# NAL-phase
# mg-nal "Na_1Mg_2(Al_5Si_1)O_12"
mnal = [1.0, 0.0, 2.5, 0.0, 2.0, 0.5];
mnal ./= sum(mnal);
# fe-nal "Na_1Fe_2(Al_5Si_1)O_12"
fnal = [1.0, 0.0, 2.5, 2.0, 0.0, 0.5];
fnal ./= sum(fnal);
# na-nal "Na_1Na_2(Al_3Si_3)O_12"
nnal = [3.0, 0.0, 1.5, 0.0, 0.0, 1.5];
nnal ./= sum(nnal);
push!(idx_of_variable_components_in_SS, find_variable_components([mnal, fnal, nnal]));
push!(idx_of_constant_components_in_SS, find_constant_components([mnal, fnal, nnal]));

nal = zeros(6);
nal[idx_of_constant_components_in_SS[end]] .= mnal[idx_of_constant_components_in_SS[end]];

#=
CONSTANTS used troughout the package
=#
# Adjust indices for concatenated vector of all SS phases
IDX_of_variable_components_in_SS = vcat([(i-1)*6 .+ idx for (i, idx) in enumerate(idx_of_variable_components_in_SS)]...);
IDX_of_variable_components_in_SS_adj = vcat([(i-1)*6 .+ idx for (i, idx) in enumerate(idx_of_variable_components_in_SS[[i for i in 1:15 if i ∉ IDX_SS_NEVER_STABLE]])]...);

# CREATE A MATRIX (6 x N_SS) OF ALL VARIABLE COMPONENTS IN SS PHASES
SS_COMP_VARIABLE = zeros(Float32, 6, (length(SS) - length(IDX_SS_NEVER_STABLE)));
for (i, j) in enumerate([idx for (i,idx) in enumerate(idx_of_variable_components_in_SS) if i ∉ IDX_SS_NEVER_STABLE])
    SS_COMP_VARIABLE[j, i] .= 1.0
end

# Concatenate all SS compositions
SS_COMP = Float32.(vcat(eval.(Symbol.(SS))...));
SS_COMP_adj = Float32.(vcat(eval.(Symbol.(SS[[i for i in 1:15 if i ∉ IDX_SS_NEVER_STABLE]]))...));

# calculate the number of variable components per ss phase
N_variable_components_in_SS = [length(v) for v in idx_of_variable_components_in_SS]
N_variable_components_in_SS_adj = [length(v) for (i, v) in enumerate(idx_of_variable_components_in_SS) if i ∉ IDX_SS_NEVER_STABLE]

# Set-up the indices of different outputs in the REG output vector
IDX_phase_frac = 1:(length(PP) + length(SS) - length(IDX_OF_PHASES_NEVER_STABLE))

# --------------------------------------------------------------------
#= FROM MAGEMin_C (these are phases from the SB24 database that are not yet fully implemented in MAGEMin)
apbo = seiferite (SiO2 in α-PbO structure)
"crst", "cristobalite"
"enm", "MgSiO3-liquid", "Mg_1Si_1O_3",
"fapv", "FeAlO3-Perovskite HS", "Fe_1Al_1O_3",
"fea", "alpha-iron", "Fe_1",
"fee", "epsilon-iron", "Fe_1",
"feg", "gamma-iron", "Fe_1",
"flpv", "fe-perovskite Low Spin", "Fe_1Si_1O_3",
"hem", "hematite", "Fe_1Fe_1O_3",
"hepv", "Fe2O3-Perovskite-HS", "Fe_1Fe_1O_3",
"hlpv", "Fe2O3-Perovskite-LS", "Fe_1Fe_1O_3",
"hmag", "High-Pressure Magnetit", "Fe_1Fe_1Fe_1O_4",
"hppv", "HS-Fe2O3-Post-Perovski", "Fe_1Fe_1O_3",
"lppv", "LS-Fe2O3-Post-Perovski", "Fe_1Fe_1O_3",
"mag", "magnetite", "Fe_1Fe_2O_4",
"mgl", "MgO-liquid", "Mg_1O_1",
"sil", "SiO2-liquid", "Si_1O_2",
"wuls", "wustite-low-spin", "Fe_2Fe_2O_4"
=#
