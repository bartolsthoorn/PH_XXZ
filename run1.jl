# Remember to set JULIA_NUM_THREADS=8
#using Pkg
#Pkg.add("Distributions")
#Pkg.add("ProgressMeter")
#Pkg.add("HDF5")

#using ProfileView
@everywhere begin
    using Distributions
    using LinearAlgebra
    using Base.Threads
    using ProgressMeter
    using HDF5

    const L = 6
    const N = 16*(L^3)

    const nsamples = 32#56
    const equilibration_mc_steps = 100000#40000

    const N_T = 17
    const N_J = 29
    const J_grid = range(-1, 0.4, length=N_J) # [-0.1]
    const T_grid = 10 .^(range(1, -3, length=N_T))

    normal_d = Normal(0,1)
    uniform_d = Uniform(0,1)

    const a1 = [1,0,0]
    const a2 = [0,1,0]
    const a3 = [0,0,1]

    const coords = 1/8. .* [[1,1,1],[5,5,1],[5,1,5],[1,5,5],
        [1,3,3],[5,3,7],[5,7,3],[1,7,7],
        [3,1,3],[3,5,7],[7,5,3],[7,1,7],
        [3,3,1],[3,7,5],[7,3,5],[7,7,1]]

    function create_random_spin_configuration()
        spins = rand(normal_d, (N,3))
        spins ./ sqrt.(sum(spins.*spins, dims=2))
    end

    coordinates = zeros(Float64, L, L, L, 16, 3)
    for lx in 1:L
        for ly in 1:L
            for lz in 1:L
                for i in 1:16
                    coordinates[lx, ly, lz, i, :] = coords[i] + (lx-1)*a1 + (ly-1)*a2 + (lz-1)*a3
                end
            end
        end
    end

    const x_local = 1/sqrt(6) .* [[-2,1,1], [-2,-1,-1], [2,1,-1], [2,-1,1]]
    const y_local = 1/sqrt(2) .* [[0,-1,1], [0,1,-1], [0,-1,-1], [0,1,1]]
    const z_local = 1/sqrt(3) .* [[1,1,1], [1,-1,-1], [-1,1,-1], [-1,-1,1]]

    function pbc(l)
        if l < 1
            l = L
        end
        if l == (L+1)
            l = 1
        end
        l
    end


    function calculate_indices(indices, inverse_indices)
        k = 1
        for lx in 1:L
            for ly in 1:L
                for lz in 1:L
                    for i in 1:16
                        indices[k,:] = [lx, ly, lz, i]
                        inverse_indices[lx, ly, lz, i] = k
                        k += 1
                    end
                end
            end
        end
    end

    indices = zeros(Int64, N, 4)
    inverse_indices = zeros(Int64, L, L, L, 16)
    calculate_indices(indices, inverse_indices)

    function create_bonds()
        bonds = zeros(Int64, 6, N)
        bond_distance = norm(coords[5]-coords[1])
        for site_i in 1:N
            lx, ly, lz, i = indices[site_i,:]
            coords_i = coordinates[lx, ly, lz, i, :]
            n = 1
            for j in 1:16
                for x in -1:1
                    for y in -1:1
                        for z in -1:1
                            d = norm(coords_i - (coords[j] + a1 * (lx+x-1) + a2 * (ly+y-1) + a3 * (lz+z-1)))
                            if abs(d - bond_distance) < 0.00001
                                bonds[n, site_i] = inverse_indices[pbc(lx+x), pbc(ly+y), pbc(lz+z), j]
                                n = n + 1
                            end
                        end
                    end
                end
            end
        end
        return bonds
    end

    function step!(spins, bonds, J, beta)
        site_i = rand(1:N)
        S1 = spins[site_i,:]
        S2 = rand(normal_d, 3)
        normalize!(S2)

        Sj = sum(spins[bonds[:, site_i],:], dims=1)

        dE = 2.0*J*(
                    (S1[1]-S2[1])*Sj[1] +
                    (S1[2]-S2[2])*Sj[2]
                ) - (S1[3]-S2[3])*Sj[3]

        if (dE <= 0) || (rand(uniform_d) < exp(-dE*beta))
            spins[site_i,:] = S2
        end
        return nothing
    end

    function mcs!(spins, bonds, J, beta)
        for i in 1:N
            step!(spins, bonds, J, beta)
        end
        return nothing
    end

    function local_to_global(spins_local)
        spins_global = zeros(Float64, N, 3)
        for k in 1:N
            lx, ly, lz, atom_i = indices[k,:]
            Sx, Sy, Sz = spins_local[k,:]
            i = (atom_i-1) ÷ 4
            spins_global[k, :] = x_local[i+1,:][1] * Sx + y_local[i+1, :][1] * Sy + z_local[i+1, :][1] * Sz
        end
        return spins_global
    end

    function mc_run(params)
        sample_step, J = params
        spins = create_random_spin_configuration()
        bonds = create_bonds()
        results = zeros(Float64, 2, N_T, N, 3)
        for (T_i, T) in enumerate(T_grid)
            beta = 1/T
            for mc_step in 1:equilibration_mc_steps
                mcs!(spins, bonds, J, beta)
            end
            results[1, T_i, :, :] = spins
            results[2, T_i, :, :] = local_to_global(spins)
        end
        return results
    end
end

function run()
    snapshots_filename = string("snapshots_", L, "_", equilibration_mc_steps, "_", nsamples, ".hdf5")
    h5open(snapshots_filename, "w") do fid
        write(fid, "coordinates", coordinates)
    end
    h5write(snapshots_filename, "J", collect(J_grid))
    h5write(snapshots_filename, "T", collect(T_grid))
    h5write(snapshots_filename, "L", L)

    results_local = zeros(Float64, N_J, N_T, nsamples, N, 3)
    results_global = zeros(Float64, N_J, N_T, nsamples, N, 3)
    #results_global = zeros(Float64, nsamples, N_J, N_T, N, 3)

    params = collect(Iterators.product(1:nsamples, J_grid))
    params_i = collect(Iterators.product(1:nsamples, 1:length(J_grid)))

    results = @showprogress 1 "Computing..." pmap(mc_run, params)
    for nsample in 1:nsamples
        for J_i in 1:length(J_grid)
            results_local[J_i,:,nsample, :, :] = results[nsample,J_i][1, :, :, :]
            results_global[J_i,:,nsample, :, :] = results[nsample,J_i][2, :, :, :]
        end
    end

    h5write(snapshots_filename, "spins_local", results_local)
    h5write(snapshots_filename, "spins_global", results_global)
end

println(N, " spins")
run()
#=
function many_step(spins, bonds, n)
    for i in 1:n
        step!(spins, bonds, -0.4, 1)
    end
    return nothing
end

function test_run()
    spins = create_random_spin_configuration()
    bonds = create_bonds()
    #@profview many_step(spins, bonds, 1)
    @time many_step(spins, bonds, 4000000)
end

test_run()
=#
#c = Condition()
#wait(c)

