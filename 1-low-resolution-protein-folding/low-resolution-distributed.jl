using Distributed
using Dates

printstyled("> Starting distributed simulation.\n"; color = :yellow)

# `machines` is a vector of machine specifications;
# Workers are started for each specification;
# A machine specification is either a string `machine_spec`
# or a tuple (machine_spec, count), where `count` is the number of worker
# processes to start on that machine and `machine_spec` is a string of the form
# [user@]host[:port] [bind_addr[:port]]
machines = Vector{Tuple{String, Int}}([ # 60/68 workers
    ("user@ip", 3),
    ("user@ip", 2),
    ("user@ip", 40),
])

printstyled("> Spawning workers.\n"; color = :yellow)
addprocs(machines,
    exename      = "julia",
    dir          = "/home/user/",
    max_parallel = 64)

# In order to generate a new path for each run, the following lines can specify
# the new path name
printstyled("> Adding current path to workers.\n"; color = :yellow)
@everywhere mkpath("/home/user/path")
@everywhere cd("/home/user/path")

printstyled("> Loading ProtoSyn.\n"; color = :yellow)

# Since, for this example, no TorchANI will se used, the environment flag
# "USE_TORCHANI" can be set to false in order to limit GPU memory allocations
@everywhere ENV["USE_TORCHANI"] = false
@everywhere using ProtoSyn
@everywhere ProtoSyn.set_logger_to_error()

# Since, for this example, the system simulated is too small to justify GPU
# usage, the ProtoSyn.SISD_0 acceleration mode should be set
@everywhere ProtoSyn.acceleration.active = ProtoSyn.SISD_0
@everywhere using ProtoSyn.Peptides
@everywhere using ProtoSyn.Calculators
@everywhere using Serialization

# Load the initial crystallographic structure from a PDB file and extract its
# sequence. Store the sequence in the cargo (to be delivered to all workers)
printstyled("> Extracting target sequence.\n"; color = :yellow)
base_truth      = Peptides.load("../data/2a3d.pdb")
sequence        = ProtoSyn.sequence(base_truth)
open("cargo/sequence.txt", "w") do file_out; write(file_out, sequence); end

# The cargo folder includes the contact map, the secondary structure map and
# sequence of the target structure to be re-generated in each worker. This
# folder is distributed to all workers via scp.
printstyled("> Transfering cargo ...\n"; color = :yellow)
for (machine, nprocs) in machines
    printstyled("  > Transfering cargo to $machine\n"; color = :yellow)
    run(`scp -r cargo $machine:$(pwd())`)
end

printstyled("> Setting up simulation environment.\n"; color = :yellow)
@everywhere begin

    # 1. Initial structure preparation
    # Generate the initial structure from aminoacid sequence. During this set-up
    # phase this pose is only used to count atoms for selection purposes (in the
    # mutators definitions)
    open("cargo/sequence.txt", "r") do file_in; global sequence = readline(file_in); end
    pose     = Peptides.build(Peptides.grammar, sequence)
    
    # Set secondary structures from the map. `ss_helix` is an AbstractSelection,
    # selection the residues with identified alpha-helix secondary structure
    ss_helix = ProtoSyn.Peptides.read_ss_map(pose, "cargo/ss_map.txt", "H")
    Peptides.setss!(pose, ProtoSyn.Peptides.SecondaryStructure[:helix], ss_helix)

    # 2. Define the energy function
    # Contact map EFC, from the contact map file
    cmap = ProtoSyn.Peptides.Calculators.Restraints.get_default_contact_restraint(filename = "../data/contact_map.txt", α = 0.1)

    # Neighbour Vector Caterpillar Solvation Energy
    sol = ProtoSyn.Peptides.Calculators.Caterpillar.get_default_caterpillar_solvation_energy(α = 0.08)
    sol.selection       = an"CA"
    sol.settings[:rmax] = 8.0
    
    # C-alpha C-alpha clash restraint
    ca_clash = ProtoSyn.Peptides.Calculators.Restraints.get_default_ca_clash_restraint(α = 10.0)
    ca_clash.settings[:d2] = 4.5
    ca_clash.settings[:d1] = 3.5
        
    energy_function = ProtoSyn.Calculators.EnergyFunction([sol, cmap, ca_clash]);

    # Since no design effort is being done, it's possible to change the dynamic
    # masks into static masks, increasing performace (ProtoSyn doesn't attempt
    # to re-calculate masks each step)
    ProtoSyn.Calculators.fixate_masks!(energy_function, pose)
    energy_function.clean_cache_every = 10000

    # Serialize/save energy function
    open(f -> serialize(f, energy_function), "energy_function.jls", "w");
    # Note: In order to load the energy function:
    # energy_function = open(deserialize, "energy_function.jls");

    # 3. Define the employed Mutators
    # Dihedral Mutator: rotates phi and psi dihedrals in non helical segments of
    # the structure by a value sampled from a gaussian distribution (centered
    # around 0.0 and with standard deviation = 0.5 rad ≈ 28.65°)
    sele_dihedral      = an"^C$|^N$"r & !ss_helix
    pmut_dihedral      = 1/count(sele_dihedral(pose))
    dihedral_mutator   = ProtoSyn.Mutators.DihedralMutator(
        randn, pmut_dihedral, 0.5, sele_dihedral)

    # Crankshaft mutator: rotates a segment of atoms between 2 C-alpha atoms by
    # a value sampled from a gaussian distribution (centered around 0.0 and with
    # standard deviation = 0.005 rad ≈ 0.3°)
    sele_crankshaft    = an"CA" & !ss_helix
    N_crankshaft       = count(sele_crankshaft(pose))
    pmut_crankshaft    = 0.5/(N_crankshaft*(N_crankshaft-1))
    crankshaft_mutator = ProtoSyn.Mutators.CrankshaftMutator(
        randn, pmut_crankshaft, 0.005, sele_crankshaft,
        !(an"^CA$|^N$|^C$|^H$|^O$"r))

    compound_mutator   = ProtoSyn.Mutators.CompoundMutator(
        [dihedral_mutator, crankshaft_mutator], nothing)

    # 4. Define the simulation Callback
    # Write the current step, total energy, temperature and acceptance ration to
    # a `log_file`, appends a pose to `out_file`. Instead of a "regular"
    # Callback, this is defined using a `setup_log` function, since each replica
    # will have to define a new Callback (the output filenames change).
    function setup_log(log_file::String, out_file::String, callback_every::Int)
        log_simulation(pose::Pose, driver_state::ProtoSyn.Drivers.DriverState) = begin
            if driver_state.step === 0
                acceptance_ratio = 1.0
            else
                acceptance_ratio = driver_state.acceptance_count / driver_state.step
            end
            s = "$(driver_state.step),$(pose.state.e[:Total]),$(driver_state.temperature),$acceptance_ratio"
            for component in energy_function.components
                s *= ",$(pose.state.e[Symbol(component.name)])"
            end
            s *= "\n"
            open(log_file, "a") do log_file_io
                write(log_file_io, s)
            end

            ProtoSyn.append(pose, out_file)
        end

        return ProtoSyn.Drivers.Callback(log_simulation, callback_every)
    end

    # 5. Define the simulation settings: 1 million steps with initial
    # temperature = 2.0 a. u.
    N_steps    = 1_000_000
    I_temp     = 2.0

    # 6. Build the monte-carlo Driver in a distributed setting. The
    # `start_simulation` consumes a job from a `job_card` `RemoteChannel` and
    # outputs any result to a `results` `RemoteChannel`. This function is
    # continuously called by all workers while there are remaining jobs
    # (i.e.: replicas).
    function start_simulation(job_cards::RemoteChannel, results::RemoteChannel)
        println("Starting simulation on worker $(myid()) - $(gethostname()) ...")
        while true
            job_n = take!(job_cards)
            println(" Consuming job $job_n on worker $(myid()) - $(gethostname())")

            # 1. Build pose from sequence, apply secondary structure and remove
            # sidechains
            pose     = Peptides.build(Peptides.grammar, sequence)
            Peptides.setss!(pose, ProtoSyn.Peptides.SecondaryStructure[:helix], ss_helix)
            ProtoSyn.Peptides.remove_sidechains!(pose, Peptides.grammar)

            # 2. Define the target output files for this replica
            log_file = "log_$job_n.csv"
            out_file = "out_$job_n.pdb"

            # 3. Define the Monte Carlo
            # The temperature used is a quadratic quench
            monte_carlo = ProtoSyn.Drivers.MonteCarlo(
                energy_function,
                compound_mutator,
                setup_log(log_file, out_file, 10000),
                N_steps,
                ProtoSyn.Drivers.get_quadratic_quench(I_temp, N_steps, 0.0))

            # 4. Initial output file definitions
            # Starts the .pdb file with the initial stretched pose and the log
            # file with the headers
            open(log_file, "w") do log_file_io
                s_title = "STEP,ENERGY,TEMPERATURE,ACCEPTANCE_RATIO"
                for component in energy_function.components
                    s_title *= ",$(uppercase(component.name))"
                end
                s_title *= "\n"
                write(log_file_io, s_title)
            end
            ProtoSyn.write(pose, out_file)
            
            # 5. Start the simulation
            monte_carlo(pose)
            info = "$job_n,$(myid()),$(gethostname())\n"
            put!(results, (pose, info))
        end
    end
end

# Populate the `job_queue` with "job cards", one for each replica.
printstyled("> Populating job queue.\n"; color = :yellow)
N         = 6000 # Number of replicas
job_queue = RemoteChannel(() -> Channel{Int16}(N))
for job_n in 1:N
    put!(job_queue, Int16(job_n))
end
results_queue = RemoteChannel(() -> Channel{Any}(N))

# Start simulation on all available workers. All workers are issued the
# `start_simulation` function, which instructs them to start taking "job cards"
# from the `job_queue` until exhaustion
printstyled("> Starting replica simulations on all workers ...\n"; color = :yellow)
for p in workers()
    remote_do(start_simulation, p, job_queue, results_queue)
end

# Save results - At this points the simulations are running. In order to more
# easily retrieve the results, the contents of the `results` RemoteChannel queue
# can be consumed and processes (in this case, written to a .pdb file). This is
# not strictly necessary, but negates the necessity to go into each worker to
# retrieve results.
function retrieve_results(pose::Pose, queue::RemoteChannel, n::Int, results_filename::String = "distributed-results.pdb")
    ProtoSyn.write(pose, results_filename)
    open(results_filename[1:(end-4)]*".txt", "w") do io_info
        Base.write(io_info, "JOB,WORKER_ID,WORKER_NAME\n")
    end

    finished_jobs = 0
    while finished_jobs < n
        _pose, _info = take!(queue)
        finished_jobs += 1
        printstyled(" > Finished jobs: $finished_jobs/$n \n"; color = :yellow)
        ProtoSyn.append(_pose, results_filename)
        open(results_filename[1:(end-4)]*".txt", "w") do io_info
            Base.write(io_info, _info)
        end
    end
end

results_filename = "distributed-results.pdb"
printstyled("> Simulations running. Saving results to $results_filename \n"; color = :yellow)
retrieve_results(pose, results_queue, N, results_filename)

printstyled("> Exiting distributed simulation.\n"; color = :yellow)

# Measure elapsed time
elapsed = now() - start
elapsed = canonicalize(Dates.CompoundPeriod(elapsed))
printstyled("> Elapsed time: $elapsed\n"; color = :red)