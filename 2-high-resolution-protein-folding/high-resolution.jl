using Printf
using ProtoSyn
using ProtoSyn.Peptides
using Serialization

# 1. Load the base truth
base_truth = Peptides.load("../data/2a3d.pdb")

# 2. Load candidate structures from the low resolution approach
res_lib = Peptides.grammar
pose    = Peptides.load("../low-resolution-fp-distributed/6_candidate.pdb")

# 3. Recover sidechains
Peptides.add_sidechains!(pose, res_lib)
ProtoSyn.write(pose, "recovered-sidechains.pdb")

# 3.1) Add default charges
ProtoSyn.Calculators.Electrostatics.assign_default_charges!(pose, res_lib)

# 4. Load the energy function components into a new energy function
all_atom_clash    = ProtoSyn.Calculators.Restraints.get_default_all_atom_clash_restraint(α = 100.0)
all_atom_clash.settings[:mask] = ProtoSyn.Calculators.get_intra_residue_mask

torchani          = ProtoSyn.Calculators.TorchANI.get_default_torchani_model(α = 2.0)
gb                = ProtoSyn.Calculators.GB.get_default_gb(α = 10.0)
sasa = ProtoSyn.Peptides.Calculators.SASA.get_default_sasa_energy(α = 0.0005)
coulomb = ProtoSyn.Calculators.Electrostatics.get_default_coulomb(α = 0.6)
hydro_bonds = ProtoSyn.Calculators.HydrogenBonds.get_default_hydrogen_bond_network(α = 0.5)
bonds = ProtoSyn.Calculators.Restraints.get_default_bond_distance_restraint(α = 100.0)

energy_function = ProtoSyn.Calculators.EnergyFunction([all_atom_clash, torchani, gb, sasa, coulomb, hydro_bonds, bonds])
ProtoSyn.Calculators.fixate_masks!(energy_function, pose)
energy_function.clean_cache_every = 1

# Save the energy function
open(f -> serialize(f, energy_function), "energy_function.jls", "w");
energy_function(pose)

# 5. Define the low resolution driver's mutators (in a compound mutator)
ss_helix = ProtoSyn.Peptides.read_ss_map(pose, "ss_map.txt", "H")

#   + Dihedral mutator
sele                                   = chi"1|2|3|4"r & !rn"PRO"
probability_mutation_dihedral_rotation = 1 / ProtoSyn.count_atoms(pose.graph)
dihedral_mutator                       = ProtoSyn.Mutators.DihedralMutator(
    randn,
    probability_mutation_dihedral_rotation,
    0.1,
    sele)

#   + Blockrot mutator
rigid_body_translation = ProtoSyn.Mutators.TranslationRigidBodyMutator(ProtoSyn.rand_vector_in_sphere, 0.1, nothing)
rigid_body_rotation = ProtoSyn.Mutators.RotationRigidBodyMutator(ProtoSyn.rand_vector_in_sphere, randn, ProtoSyn.center_of_mass, 0.01, nothing)
sele = ProtoSyn.RandomSelectionFromList([rid"2:20", rid"27:44", rid"50:71"])
rigid_body = ProtoSyn.Mutators.CompoundMutator([rigid_body_translation, rigid_body_rotation], sele)

sd_energy_function                   = ProtoSyn.Calculators.EnergyFunction([torchani, bonds])
sd_energy_function.clean_cache_every = CLEAN_CACHE_EVERY
sd_energy_function.update_forces     = true

steepest_descent = ProtoSyn.Drivers.SteepestDescent(sd_energy_function, nothing, 500, 0.01, 0.5)

blockrot_mutator = ProtoSyn.Drivers.CompoundDriver([rigid_body, steepest_descent], 0.20)

#   + Rotamer mutator
rotamer_library = ProtoSyn.Peptides.load_dunbrack()
probability_mutation = 1 / ProtoSyn.count_residues(pose.graph)
rotamer_mutator      = ProtoSyn.Peptides.Mutators.RotamerMutator(rotamer_library, probability_mutation, 5, !rn"PRO", true)

compound_mutator   = ProtoSyn.Drivers.CompoundDriver(
    [blockrot_mutator, dihedral_mutator, rotamer_mutator])

# 6. Define a custom callback
function setup_log(log_file::String, out_file::String)
    return ProtoSyn.Common.default_energy_step_frame_detailed(5, out_file, "Candidate 6", :cyan, log_file, true)
end

# 7. Define the low resolution driver (Monte Carlo)
N_steps          = 500
I_temp           = 6.0
N                = 5 # Number of replicas
results_filename = "high-res-results.pdb"

# 8. Start the simulation
printstyled(" > Starting simulation ... \n"; color = :yellow)
ProtoSyn.write(pose, results_filename)
ProtoSyn.set_logger_to_error()

for replica in 1:N
    log_file = "log_$replica.csv"
    out_file = "out_$replica.pdb"

    # 8.1) Define the starting point for each replica
    local pose = Peptides.load("../low-resolution-fp-distributed/6_candidate.pdb")
    Peptides.add_sidechains!(pose, res_lib)
    ProtoSyn.pop_atoms!(pose, as"H")
    ProtoSyn.add_hydrogens!(pose, res_lib)
    ProtoSyn.Peptides.assign_default_atom_names!(pose)
    ProtoSyn.Calculators.Electrostatics.assign_default_charges!(pose, res_lib)
    sync!(pose)

    open(log_file, "w") do log_file_io
        write(log_file_io, "STEP,ENERGY,TEMPERATURE,ACCEPTANCE_RATIO,TORCHANI,SOLVATION,BACKBONE_CLASH,ALL_ATOM_CLASH,CONTACT\n")
    end

    # 8.2) Define the monte carlo driver for each replica (only now, since the 
    # filenames change for each replica)
    monte_carlo = ProtoSyn.Drivers.MonteCarlo(
        energy_function,
        compound_mutator,
        setup_log(log_file, out_file),
        N_steps,
        ProtoSyn.Drivers.get_quadratic_quench(I_temp, N_steps, 0.0))

    # 8.3) Run the simulation
    ProtoSyn.write(pose, out_file)
    steepest_descent(pose)
    monte_carlo(pose)
    ProtoSyn.append(pose, results_filename)
    printstyled(" > Finished jobs: $replica/$N \n"; color = :yellow)
end